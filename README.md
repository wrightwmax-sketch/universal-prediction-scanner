# Universal Prediction Market AI Edge Scanner (`universal-prediction-scanner`)

A zero-server, autonomous cross-exchange prediction market scanner and interactive GitHub Pages dashboard.

## Overview

`universal-prediction-scanner` ingests, normalizes, and evaluates binary prediction markets across major platforms via their public, unauthenticated read APIs:

1. **Polymarket** (Global Crypto CLOB — Gamma API + CLOB Order Book API)
2. **Kalshi** (US CFTC-Regulated Exchange — Public Trade API v2 Markets + Orderbook)
3. **Manifold Markets** (Tech/Culture/Niche AMM — Public API v0)
4. **Metaculus** (Calibrated Superforecaster Benchmark — Public API v2)

## Core Methodology

1. **Historical Settlement Calibration**: Pulls recently settled markets from Polymarket (`closed=true`) and Kalshi (`status=settled`) to ground base rates and resolution quirks.
2. **True Executable Ask Price Extraction**:
   - **Polymarket**: Queries `https://clob.polymarket.com/book?token_id={TOKEN_ID}` for best executable Ask on YES and NO tokens.
   - **Kalshi**: Reads `yes_ask` and `no_ask` (in cents) from `/trade-api/v2/markets`.
   - **Manifold**: Applies a +2¢ AMM slippage buffer to implied probabilities.
3. **Blind LLM Evaluation with Live Google Search Grounding**:
   - Uses `google-genai` (`gemini-3.8-flash` with `google_search` grounding tool and structured Pydantic output) **without** exposing current market prices in the prompt to eliminate price anchoring.
   - Performs a Pre-Mortem Scenario Fault-Tree and audits exact platform resolution fine print (UMA Optimistic Oracle rules, CFTC contract specifications, and creator resolution rules).
4. **Strict High-Conviction Filtering**:
   - True Win Probability (`llm_true_probability`) $\ge 0.82$ ($82\%$)
   - Epistemic Confidence (`confidence_score`) $\ge 0.75$ ($75\%$)
   - Net Edge over Executable Ask Price (`llm_true_probability - executable_ask_price`) $\ge 0.08$ ($+8\text{¢}$)

## Repository Structure

```text
universal-prediction-scanner/
├── .github/
│   └── workflows/
│       └── scan_markets.yml
├── requirements.txt
├── scanner.py
├── signals.json
├── index.html
└── README.md
```

## Running Locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-api-key"
python scanner.py
```

## Automated GitHub Actions & GitHub Pages

1. Add `GEMINI_API_KEY` to your repository's **Settings → Secrets and variables → Actions**.
2. Enable **GitHub Pages** on the `main` branch (`/` root).
3. `.github/workflows/scan_markets.yml` runs every 4 hours (`0 */4 * * *`) or on manual `workflow_dispatch`, updating `signals.json` in place.
