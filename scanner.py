import os
import json
import time
from datetime import datetime, timezone
import requests
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# API Base URLs
POLY_GAMMA_URL = "https://gamma-api.polymarket.com"
POLY_CLOB_URL = "https://clob.polymarket.com"
KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2"
MANIFOLD_API_URL = "https://api.manifold.markets/v0"
METACULUS_API_URL = "https://www.metaculus.com/api2"

# Model configuration (substituted gemini-3.8-flash for deprecated gemini-2.5-pro)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

# Tier 1 Thresholds: High-Conviction Safe Compounders
MIN_WIN_PROBABILITY = 0.82
MIN_CONFIDENCE_SCORE = 0.75
MIN_NET_EDGE = 0.08

# Tier 2 Thresholds: Educated-Risk / Asymmetric High-Payout Multipliers (2.2x - 8.0x+)
ASYMMETRIC_MIN_ASK = 0.12
ASYMMETRIC_MAX_ASK = 0.45
ASYMMETRIC_MIN_TRUE_PROB = 0.35
ASYMMETRIC_MIN_NET_EDGE = 0.12
ASYMMETRIC_MIN_ROI_PCT = 35.0

MAX_TOTAL_CANDIDATES = 24

class ScenarioBranch(BaseModel):
    scenario_name: str = Field(description="Specific possibility or failure mode before deadline")
    probability: float = Field(description="Estimated probability of this branch (0.0 to 1.0)")
    impact_on_resolution: str = Field(description="Whether this resolves YES or NO under platform rules")

class MarketEvaluation(BaseModel):
    plain_english_summary: str = Field(description="2-sentence high-school level explanation of why this bet wins and why the crowd is wrong")
    historical_precedents: list[str] = Field(description="Historical base rates and past similar events found via search and resolved markets")
    resolution_rule_traps: str = Field(description="Exact definitions, loopholes, or edge cases in the platform's resolution criteria")
    scenario_tree: list[ScenarioBranch] = Field(description="Exhaustive fault-tree of possibilities that could occur before expiration")
    true_yes_probability: float = Field(description="Calibrated true probability of YES resolving (0.01 to 0.99), blinding out current market price")
    true_no_probability: float = Field(description="Calibrated true probability of NO resolving (0.01 to 0.99)")
    confidence_score: float = Field(description="Epistemic confidence in public evidence (0.0 to 1.0)")

def fetch_historical_calibration() -> str:
    lines = []
    try:
        r = requests.get(f"{POLY_GAMMA_URL}/markets", params={"closed": "true", "limit": 10, "order": "volumeNum", "ascending": "false"}, timeout=12)
        if r.ok:
            for m in r.json()[:8]:
                lines.append(f"[Polymarket Settled] {m.get('question')} -> Outcome Prices: {m.get('outcomePrices')}")
    except Exception:
        pass
    try:
        rk = requests.get(f"{KALSHI_API_URL}/markets", params={"status": "settled", "limit": 10}, timeout=12)
        if rk.ok:
            for m in rk.json().get("markets", [])[:8]:
                lines.append(f"[Kalshi Settled] {m.get('title')} -> Result: {m.get('result')}")
    except Exception:
        pass
    return "\n".join(lines) if lines else "Standard base rate: binary hype deadlines resolve NO ~83% of the time unless hard regulatory/technical prerequisites are already filed."

def get_poly_best_ask(token_id: str) -> float | None:
    try:
        r = requests.get(f"{POLY_CLOB_URL}/book", params={"token_id": token_id}, timeout=8)
        asks = r.json().get("asks", [])
        return min(float(a["price"]) for a in asks) if asks else None
    except Exception:
        return None

def fetch_polymarket_candidates(limit: int = 10) -> list[dict]:
    out = []
    try:
        r = requests.get(f"{POLY_GAMMA_URL}/markets", params={"active": "true", "closed": "false", "limit": 100, "order": "volumeNum", "ascending": "false"}, timeout=15)
        for m in r.json():
            vol = float(m.get("volumeNum", 0) or 0)
            if vol < 20000:
                continue
            clob_raw = m.get("clobTokenIds")
            outcomes_raw = m.get("outcomes")
            if not clob_raw or not outcomes_raw:
                continue
            token_ids = json.loads(clob_raw) if isinstance(clob_raw, str) else clob_raw
            outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
            if len(token_ids) != 2 or [o.upper() for o in outcomes] != ["YES", "NO"]:
                continue
            yes_ask = get_poly_best_ask(token_ids[0])
            no_ask = get_poly_best_ask(token_ids[1])
            if yes_ask is None or no_ask is None:
                continue
            if (yes_ask <= 0.03 or yes_ask >= 0.96) and (no_ask <= 0.03 or no_ask >= 0.96):
                continue
            events = m.get("events") or []
            slug = events[0].get("slug") if events and isinstance(events, list) and events[0].get("slug") else m.get("slug", "")
            out.append({
                "platform": "Polymarket",
                "question": m.get("question"),
                "description": m.get("description", ""),
                "resolution_source": m.get("resolutionSource", "UMA Optimistic Oracle"),
                "end_date": m.get("endDate"),
                "volume_usd": vol,
                "url": f"https://polymarket.com/event/{slug}",
                "yes_ask": yes_ask,
                "no_ask": no_ask
            })
            if len(out) >= limit:
                break
    except Exception as e:
        print(f"[WARN] Polymarket ingest error: {e}")
    return out

def fetch_kalshi_candidates(limit: int = 8) -> list[dict]:
    out = []
    try:
        r = requests.get(f"{KALSHI_API_URL}/events", params={"status": "open", "with_nested_markets": "true", "limit": 100}, timeout=15)
        for ev in r.json().get("events", []):
            series = (ev.get("series_ticker") or ev.get("event_ticker") or "").lower()
            ev_title = ev.get("title") or ""
            for m in ev.get("markets", []):
                if m.get("mve_collection_ticker"):
                    continue
                vol = float(m.get("volume_fp") or m.get("volume") or 0)
                if vol < 500:
                    continue
                ya_d = m.get("yes_ask_dollars")
                na_d = m.get("no_ask_dollars")
                ya_c = m.get("yes_ask")
                na_c = m.get("no_ask")
                yes_ask = float(ya_d) if ya_d is not None else (float(ya_c) / 100.0 if ya_c is not None else 0.0)
                no_ask = float(na_d) if na_d is not None else (float(na_c) / 100.0 if na_c is not None else 0.0)
                if not (0.03 < yes_ask < 0.96 and 0.03 < no_ask < 0.96):
                    continue
                ticker = m.get("ticker", "")
                sub = m.get("yes_sub_title") or m.get("subtitle") or ""
                question = f"{ev_title} ({sub})" if sub and sub.lower() not in ev_title.lower() else (m.get("title") or ev_title or ticker)
                out.append({
                    "platform": "Kalshi",
                    "question": question,
                    "description": f"{m.get('rules_primary', '')} {m.get('rules_secondary', '')}".strip(),
                    "resolution_source": "CFTC Regulated Kalshi Contract Spec",
                    "end_date": m.get("close_time") or m.get("expiration_time"),
                    "volume_usd": vol,
                    "url": f"https://kalshi.com/markets/{series}",
                    "yes_ask": yes_ask,
                    "no_ask": no_ask
                })
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
    except Exception as e:
        print(f"[WARN] Kalshi ingest error: {e}")
    return out

def fetch_manifold_candidates(limit: int = 6) -> list[dict]:
    out = []
    try:
        r = requests.get(f"{MANIFOLD_API_URL}/search-markets", params={"filter": "open", "contractType": "BINARY", "sort": "liquidity", "limit": 25}, timeout=15)
        for m in r.json():
            prob = m.get("probability")
            if prob is None or not (0.06 < float(prob) < 0.94):
                continue
            yes_ask = min(0.96, float(prob) + 0.02)
            no_ask = min(0.96, (1.0 - float(prob)) + 0.02)
            close_ms = m.get("closeTime")
            end_iso = datetime.fromtimestamp(close_ms / 1000.0, tz=timezone.utc).isoformat() if close_ms else "Open"
            out.append({
                "platform": "Manifold",
                "question": m.get("question"),
                "description": str(m.get("textDescription", ""))[:1500],
                "resolution_source": "Manifold Market Creator & Moderator Rules",
                "end_date": end_iso,
                "volume_usd": float(m.get("volume", 0) or 0),
                "url": m.get("url", "https://manifold.markets"),
                "yes_ask": yes_ask,
                "no_ask": no_ask
            })
            if len(out) >= limit:
                break
    except Exception as e:
        print(f"[WARN] Manifold ingest error: {e}")
    return out

def evaluate_market_blind(client: genai.Client, market: dict, historical_context: str) -> MarketEvaluation | None:
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    prompt = f"""
You are an elite cross-exchange quantitative prediction market analyst. Estimate the TRUE probability of this binary contract resolving YES vs NO without knowing the current market price.

CURRENT UTC TIME: {now_iso}

HISTORICAL MULTI-EXCHANGE CALIBRATION:
{historical_context}

TARGET CONTRACT:
- Exchange Platform: {market['platform']}
- Question / Contract Title: {market['question']}
- Expiration / Close Date: {market['end_date']}
- Settlement Authority: {market['resolution_source']}
- Exact Contract Rules & Fine Print:
{market['description']}

MANDATORY STEPS:
1. LIVE WEB & DOCKET SEARCH: Search the live internet for primary evidence (SEC EDGAR filings, FCC OET grants, federal court dockets, BLS/Fed releases, polling aggregators, and historical base rates).
2. AUDIT PLATFORM FINE PRINT: Check whether an "announcement", "beta", or "preliminary reading" counts, or if final official publication/commercial shipping is required by the exchange rules.
3. EXHAUSTIVE FAULT-TREE: List every realistic scenario branch between now and expiration, ensuring branch probabilities sum to 1.0.
4. OUTPUT CALIBRATED PROBABILITIES: Remember that dramatic deadline contracts resolve NO ~83% of the time unless concrete prerequisite steps are already completed and verifiable.
"""
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_mime_type="application/json",
                response_schema=MarketEvaluation,
                temperature=0.1,
            ),
        )
        return MarketEvaluation.model_validate_json(response.text)
    except Exception as e:
        print(f"[ERROR] Evaluation failed for [{market['platform']}] {market['question']}: {e}")
        return None

def classify_side(prob: float, ask: float, confidence: float) -> str | None:
    if ask <= 0.03 or ask >= 0.96 or confidence < MIN_CONFIDENCE_SCORE:
        return None
    edge = prob - ask
    roi_pct = (edge / ask) * 100.0
    # Tier 1: High-Conviction Safe Compounder (>=82% win probability, >=+8c edge)
    if prob >= MIN_WIN_PROBABILITY and edge >= MIN_NET_EDGE:
        return "SAFE_COMPOUNDER"
    # Tier 2: Educated-Risk High-Payout Multiplier (12c-45c ask = 2.2x-8.3x payout, >=35% true prob, >=+12c edge, >=35% EV)
    if ASYMMETRIC_MIN_ASK <= ask <= ASYMMETRIC_MAX_ASK and prob >= ASYMMETRIC_MIN_TRUE_PROB and edge >= ASYMMETRIC_MIN_NET_EDGE and roi_pct >= ASYMMETRIC_MIN_ROI_PCT:
        return "EDUCATED_RISK_HIGH_PAYOUT"
    return None

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[WARN] GEMINI_API_KEY not set. Preserving existing signals.json.")
        return

    client = genai.Client(api_key=api_key)
    historical_context = fetch_historical_calibration()

    candidates = (
        fetch_polymarket_candidates(limit=10)
        + fetch_kalshi_candidates(limit=8)
        + fetch_manifold_candidates(limit=6)
    )[:MAX_TOTAL_CANDIDATES]

    signals = []
    educated_risk_signals = []
    all_studied = []

    for m in candidates:
        eval_res = evaluate_market_blind(client, m, historical_context)
        if not eval_res:
            continue

        matched_any = False
        for side, prob, ask in [
            ("YES", eval_res.true_yes_probability, m["yes_ask"]),
            ("NO", eval_res.true_no_probability, m["no_ask"])
        ]:
            tier = classify_side(prob, ask, eval_res.confidence_score)
            if tier:
                matched_any = True
                edge = prob - ask
                roi_pct = (edge / ask) * 100.0
                payout_mult = round(1.0 / ask, 2)
                q_kelly = 0.25 * max(0.0, (prob - ask) / (1.0 - ask)) * 100.0
                rec = {
                    "platform": m["platform"],
                    "question": m["question"],
                    "url": m["url"],
                    "recommended_side": side,
                    "strategy_tier": tier,
                    "status": "QUALIFIED_HIGH_CONVICTION" if tier == "SAFE_COMPOUNDER" else "EDUCATED_RISK_HIGH_PAYOUT",
                    "executable_ask_price": round(ask, 3),
                    "payout_multiplier": payout_mult,
                    "llm_true_probability": round(prob, 3),
                    "likelihood_to_convert_pct": round(prob * 100, 1),
                    "net_edge_cents": round(edge * 100, 1),
                    "expected_roi_pct": round(roi_pct, 1),
                    "quarter_kelly_bankroll_pct": round(q_kelly, 1),
                    "recommended_bankroll_stake_pct": 20.0 if tier == "SAFE_COMPOUNDER" else 5.0,
                    "confidence_score": round(eval_res.confidence_score, 2),
                    "volume_usd": round(m["volume_usd"], 0),
                    "end_date": m["end_date"],
                    "plain_english_summary": eval_res.plain_english_summary,
                    "resolution_rule_traps": eval_res.resolution_rule_traps,
                    "historical_precedents": eval_res.historical_precedents,
                    "scenario_tree": [s.model_dump() for s in eval_res.scenario_tree]
                }
                all_studied.append(rec)
                if tier == "SAFE_COMPOUNDER":
                    signals.append(rec)
                else:
                    educated_risk_signals.append(rec)

        if not matched_any:
            best_side = "YES" if (eval_res.true_yes_probability - m["yes_ask"]) >= (eval_res.true_no_probability - m["no_ask"]) else "NO"
            best_prob = eval_res.true_yes_probability if best_side == "YES" else eval_res.true_no_probability
            best_ask = m["yes_ask"] if best_side == "YES" else m["no_ask"]
            best_edge = best_prob - best_ask
            best_roi = (best_edge / best_ask) * 100.0 if best_ask > 0 else 0.0
            best_kelly = 0.25 * max(0.0, (best_prob - best_ask) / (1.0 - best_ask)) * 100.0 if best_ask < 1.0 else 0.0
            all_studied.append({
                "platform": m["platform"],
                "question": m["question"],
                "url": m["url"],
                "recommended_side": best_side,
                "strategy_tier": "WATCHLIST",
                "status": "WATCHLIST_NEAR_MISS" if best_edge > 0 else "REJECTED_TRAP_ALERT",
                "executable_ask_price": round(best_ask, 3),
                "payout_multiplier": round(1.0 / best_ask, 2) if best_ask > 0 else 0.0,
                "llm_true_probability": round(best_prob, 3),
                "likelihood_to_convert_pct": round(best_prob * 100, 1),
                "net_edge_cents": round(best_edge * 100, 1),
                "expected_roi_pct": round(best_roi, 1),
                "quarter_kelly_bankroll_pct": round(best_kelly, 1),
                "recommended_bankroll_stake_pct": 0.0,
                "confidence_score": round(eval_res.confidence_score, 2),
                "volume_usd": round(m["volume_usd"], 0),
                "end_date": m["end_date"],
                "plain_english_summary": eval_res.plain_english_summary,
                "resolution_rule_traps": eval_res.resolution_rule_traps,
                "historical_precedents": eval_res.historical_precedents,
                "scenario_tree": [s.model_dump() for s in eval_res.scenario_tree]
            })
        time.sleep(2)

    signals.sort(key=lambda x: x["expected_roi_pct"], reverse=True)
    educated_risk_signals.sort(key=lambda x: x["expected_roi_pct"], reverse=True)
    all_studied.sort(key=lambda x: x["expected_roi_pct"], reverse=True)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    payload = {
        "last_updated_utc": now_str,
        "platforms_scanned": ["Polymarket", "Kalshi", "Manifold"],
        "total_markets_scanned": len(all_studied),
        "high_conviction_count": len(signals),
        "educated_risk_count": len(educated_risk_signals),
        "signals": signals,
        "educated_risk_signals": educated_risk_signals,
        "all_studied_opportunities": all_studied
    }
    with open("signals.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open("pulse_check_study.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

if __name__ == "__main__":
    main()
