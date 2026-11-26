# Plan: Turn the BTCUSDT Trading Bot into a Sellable AI Agent Skill (Agensi.io)

> Status: Draft plan · Created after full code analysis of `trading-bot/`
> Target marketplace: https://www.agensi.io/sell · Category: Trading & Investing / Web3 & Crypto
> Last updated: 2026-11-23 — synced with `NEXT_STEPS.md` after Phase 3 completion (`221b753`, `92e41bd`)

---

## 1. Executive summary

| Question | Answer |
|---|---|
| What was analysed | Python BTCUSDT futures trading bot (`trading-bot/`), ~13,400 LOC across 12 modules + 8 test files, dual-ML (CatBoost tactical + strategic) architecture |
| What it does | Buys/sells BTCUSDT perpetual futures from an ML-driven signal: tactical model predicts 1h-ahead return from 15m candles, strategic model picks stake/SL/TP/leverage/regime |
| Key differentiators | Walk-forward retraining, regime detection, adaptive threshold/liquidity/trailing logic, multi-layer risk management, position adoption, hot model swap |
| Proposed skill | Package the **strategy engineering playbook** as a reusable agent skill ("BTC Dual-ML Strategy") |
| Target buyer | Developer/quant who wants an AI agent to design + backtest a dual-ML crypto trading strategy without starting from zero |
| Revenue model | One-time $29 (launch) → subscription $19–29/mo (updates); creator keeps **70%** of net |

---

## 2. Bot analysis (condensed — full detail in README.md / NEXT_STEPS.md)

### 2.1 Architecture

| Module | Responsibility |
|---|---|
| `main.py` | CLI: `train`, `simulate`, `live`, `compare` |
| `config.py` | All hyperparameters + env overrides + testnet/mainnet switch |
| `data.py` | Binance klines download, feature engineering, labeling, regime/abs-vol classification, adaptive thresholds |
| `model.py` | CatBoost tactical model, walk-forward `rolling_tactical_predict`, strategic multi-output meta-params |
| `broker.py` | Binance REST client, rate limiter (1000 w/min), brackets (TP/SL), scale-in, partial close, position cache |
| `orderbook.py` | WebSocket orderbook recorder → liquidity gates (spread, depth, vanish, aggressor, intensity) |
| `strategy.py` | `DualMLStrategy`: signal → gates → entry/exit/scale-in/partial logic |
| `simulate.py` | `MockBroker` backtest + metrics (return, Sharpe, drawdown, win rate, profit factor) |
| `compare.py` | Sim vs demo comparison + HTML report |
| `logger.py`, `utils.py` | Logging, CSV/equity persistence |

### 2.2 ML & signal design

- **Tactical ML**: CatBoost regressor, 15m candles → predicts `future_ret` ~4 candles ahead (1h horizon). Walk-forward retrain cadence configurable (sim: every candle; live: every N iterations).
- **Strategic ML**: multi-output CatBoost on 1h features → meta params: `recommended_leverage`, `max_exposure_frac`, `stake_long_frac`, `stake_short_frac`, `stop_loss_frac`, `take_profit_frac`, `max_hold_hours`, plus `regime` (trend/chop/high_vol/halt).
- **Signal**: `pred > adaptive_threshold → long`, `pred < -thr → short`, else hold. Adaptive threshold = 95th pct of last 200 \|preds\], clamped [0.002, 0.02], fallback 0.006.

### 2.3 Risk / execution layers (the "expertise" to encode in the skill)

- Entry gates: extreme-vol gate (vol_ratio > 1.8), absolute-volume gate (abs_vol_ratio < 0.10), liquidity gate (spread ≤ 10 bps, depth ≥ 50k USD, vanish ≤ 60%, aggressor ratio), chop-regime gate, spread-aware entry gate, 30-min trade cooldown.
- Exits: max-hold, SL/TP, ATR trailing stop (breakeven after 1R), time-weighted SL tightening (75% → progressive to 50% of SL at max hold), liquidity emergency exit, reversal handling (partial 33% → full close + flip after 2 reversals).
- Profit taking: R-multiple partial exits (1R/2R/3R, fractions 33/33/34).
- Position sizing: confidence-weighted stake (sqrt scaling, cap 3×), scale-ins (2 confirm bars, max 2, min 0.5R profit).
- Ops: position adoption on restart, hot model swap, graceful shutdown close, rate limits/circuit breakers, feature cache, strategy decision logging.

### 2.4 Current state (updated Nov 23, 2026 — commit `92e41bd`)

- **Phase 3 improvements marked completed** (`221b753 Implement all suggested improvements` + follow-up `92e41bd Update NEXT_STEPS.md`):
  - Time-weighted SL: trigger moved from 50% → **75% of `max_hold_hours`** (`TIME_WEIGHTED_SL_TRIGGER=0.75`) with progressive tightening (`TIME_WEIGHTED_SL_PROGRESSIVE=True`).
  - Trailing SL: immediate trail from entry (`TRAILING_START_IMMEDIATE=True`); floor at initial SL until 1R profit, then breakeven.
  - R-multiple profit taking: partial exits at 1R/2R/3R (33%/33%/34%), triggered at 0.95R/1.95R/2.95R (5% buffer); remainder protected by trailing SL.
  - Scale-in: minimum **0.5R** profit required; scale stake fraction **33%**.
- Still open: dynamic position sizing by volatility, trailing-SL refinements (min distance / chandelier exit), liquidity-gate refinements, cost-to-edge tracking, monitoring/alerting, automated retraining pipeline.
- No backtest metrics (`logs/`, `models/`) are committed — actual performance numbers would come from the owner’s local runs.

---

## 3. Proposed skill concept

### 3.1 Recommended product: `btc-dual-ml-strategy` (BTC Dual-ML Trading Strategy)

**One-liner:** *An AI agent skill that designs, backtests and validates dual-ML (tactical + strategic) trading strategies for BTCUSDT crypto futures.*

**Buyer pain:** Building a robust crypto momentum/trend bot is hard — feature engineering, labeling, CatBoost, walk-forward validation, risk layers, backtest metrics. Most developers either copy a template blindly or give up after one failed backtest.

**Skill outcome (what the buying agent can do after purchase):**
1. Turn a trading hypothesis into a dual-ML strategy design doc.
2. Engineer the feature set (returns/lags, EMA, ATR, volatility, cyclical time, regime, abs-vol).
3. Label with walk-forward `future_ret` horizon.
4. Train tactical + strategic CatBoost models.
5. Run walk-forward validation + backtest with realistic fees/slippage.
6. Apply the full risk stack (gates, trailing, time-weighted SL, partial/R exits, scale-in, cooldown).
7. Produce a structured backtest report (return, Sharpe, drawdown, win rate, profit factor).

**Why it should sell:**
- Deep domain expertise (the owner has lived this: multiple post-mortems, adaptive gates, fixes).
- Comprehensive scope → premium positioning ($20–50 tier per Agensi guidance).
- Narrow enough to be agent-executable (one thing well).
- Category fit: `/skills?category=trading-investing` and Web3 & Crypto.

**Honesty & scope guardrails (critical for Agensi’s AI reviewer):**
- Skill encodes strategy *design/backtest* workflow — it does **not** place live trades or hold API keys.
- Must include risk disclaimer: backtests are historical, live trading involves substantial risk, not financial advice.

### 3.2 Target listing metadata (draft)

- **Skill name:** `btc-dual-ml-strategy` (folder name must match exactly; lowercase + hyphens).
- **Description (draft):** “Designs, backtests, and validates dual-ML (tactical plus strategic machine learning) trading strategies for BTCUSDT crypto futures. Use when the user wants to build a Bitcoin trading bot, evaluate a momentum or trend strategy, run walk-forward backtests, or design risk management and position sizing for a crypto trading system.”
- **Tags:** crypto, bitcoin, btc, trading, backtesting, machine-learning, catboost, risk-management, algo-trading, futures.
- **Category:** Trading & Investing (fallback: Web3 & Crypto).

---

## 4. Skill creation plan (package → test → submit)

### 4.1 Folder layout (zip target)

```
btc-dual-ml-strategy/
├── SKILL.md
└── references/
    ├── strategy-rules.md      # thresholds, gates, exits, sizing — quick reference tables
    └── feature-labeling.md    # feature definitions, label horizon, regime logic
```

### 4.2 SKILL.md content plan (body structure)

```markdown
# BTC Dual-ML Trading Strategy

## When to use this skill
- User asks to build / design a Bitcoin trading bot for BTCUSDT
- User wants to evaluate or backtest a momentum / trend crypto strategy
- User asks for walk-forward validation or ML-based trading signals
- User asks for crypto position sizing / risk management design

## When NOT to use
- User wants live trade execution instructions or broker/API integration help
- User wants financial advice / market predictions

## Process (numbered)
1. Clarify hypothesis & market universe (BTCUSDT perpetuals, futures sizing).
2. Engineer features: ret1, ret_lag_*, EMA_*, atr14, vol_12/48, cyclical time, regime, abs_vol_ratio.
3. Label: future_ret with horizon = 4 × timeframe (1h ahead for 15m bars).
4. Train tactical CatBoost (15m → future return) + strategic multi-output CatBoost (1h → meta params).
5. Walk-forward validation; reject degenerate models (< MIN_TREES).
6. Backtest via MockBroker-equivalent logic; compute return, Sharpe, max DD, win rate, profit factor; net of fee 0.0004 / slippage 0.0003.
7. Apply risk stack: entry gates, trailing stop, time-weighted SL, partial/R exits, scale-in, cooldown.
8. Iterate: compare live vs sim, hot-swap, adopt position.

## Decision rules / quick reference
[condensed tables from config.py]

## Output format
Strategy design doc + backtest report + next-iteration checklist.

## Boundaries
- Does not execute trades or require broker credentials.
- Not financial advice; historical backtests ≠ future results.
```

### 4.3 Supporting references

- `references/strategy-rules.md` — full constants: thresholds (ABSOLUTE_THRESHOLD 0.006, adaptive bounds), stake fractions, SL/TP, max hold, leverage map, regime thresholds (trend_strength 0.4, vol_ratio 1.4), gates (extreme vol 1.8, abs-vol 0.10, liq spread/depth/vanish/aggressor/intensity), trailing (ATR mult, breakeven), time-weighted SL (0.75 trigger → progressive), R-partial exits, scale-in, cooldown.
- `references/feature-labeling.md` — feature list, label horizon, regime classification, abs-vol override to `halt`.

### 4.4 Agensi compliance checklist (mapped to their 8 security checks)

1. **File structure validation** — zip must contain SKILL.md at root (or in skill folder); references/ OK.
2. **File type screening** — only `.md`, `.py`(safe helpers?), text; no binaries.
3. **Dangerous command pattern detection** — no shell exploits, no `curl|sh`, no destructive commands. (If scripts are added, keep pure Python functions without subprocess/network.)
4. **Secrets detection** — must NOT include Binance API keys, `.env` content, tokens. → Scrub all credentials; the skill must not instruct agents to obtain/steal keys.
5. **Environment variable harvesting analysis** — do not instruct agents to read/collect env vars or prompt users for secrets.
6. **Network access auditing** — no hidden network calls; reference Binance REST endpoints only as documented data-source context (or omit).
7. **Obfuscation detection** — plain, readable markdown/Python.
8. **Prompt injection screening** — no instructions to exfiltrate context or override user instructions.

### 4.5 Test-before-publish (per Agensi guide)

- Trigger testing: sample prompts ("build me a BTC trading bot", "backtest a momentum strategy for BTC") → confirm reliable activation.
- Output quality: run skill against a toy strategy → verify structured output quality.
- Cross-agent compatibility: use only `name`, `description`, optional `when_to_use`, plain markdown body → works in Claude Code, Codex CLI, Cursor, OpenCode, OpenClaw, Gemini CLI.

### 4.6 Submission sequence

1. Dashboard → Settings → enable **Creator Mode**.
2. Creator Studio → Payouts → connect **Stripe** (or **Zoneless/USDC** if Stripe unavailable).
3. Build zip, upload via **Creator Dashboard → Submit a Skill**.
4. Set price (launch: **$29** one-time), tags, listing description.
5. Submit → automated 8-point scan → manual review (expect 24–48h).
6. If approved → live on marketplace; creator page at `agensi.io/creators/<handle>`.

---

## 5. Pricing & payout plan

| Tier | Price | Buyer gets | Creator keeps (70% of net) | Economics per sale |
|---|---|---|---|---|
| One-time launch | **$29** | Full skill + references | $20.30 | 100 sales → $2,030 |
| Subscription | $19–29/mo | Skill + future updates | 70% of net/mo | — |

- Paid skills require connected payout method before going live; waiting earnings threshold **$3.50**; Stripe → bank at sale; Zoneless → USDC 14 days after sale.
- Launch recommendation: **$29 one-time** (comprehensive deep-domain skill). Add subscription tier later once live updates/v2 exist.
- Free tier option: release a small free variant (e.g., `btc-regime-classifier`) later to funnel into the paid skill — free listings need no payout method.

---

## 6. Go-to-market / marketing plan

### 6.1 Pre-launch (weeks 1–2)
- Finalize `SKILL.md` + references; run trigger tests; assemble zip.
- Creator profile: bio, avatar, socials (GitHub/X/LinkedIn), pin best listing.

### 6.2 Launch (week 3)
- Submit for review; set price/tags/description.
- Go live after approval; claim marketplace URL; add listing to GitHub repo (docs/ or README section).

### 6.3 Post-launch (weeks 4–12)
- **Week 4:** launch announcement — X thread + LinkedIn post showing before/after: "messy manual strategy research → agent running dual-ML workflow in one prompt"; link to Agensi listing.
- **Week 5–6:** short-form UGC screen recording of agent using the skill (TikTok/Reels/Shorts/X) — Agensi reports this outperforms static posts.
- **Week 7–8:** dev.to / Reddit (r/algotrading, r/quant, r/MachineLearning) build-in-public post; share backtest numbers from owner's runs.
- **Ongoing:** ship v1.1 updates (reflect remaining NEXT_STEPS.md work) → announce → existing buyers auto-get updates.

### 6.4 Positioning angles for the listing
- "The Bitcoin trading bot, packaged as an AI skill."
- "From hypothesis to validated backtest — in one agent prompt."
- Compare with generic crypto trading skills (lack dual-ML rigor).

---

## 7. Risks & compliance notes

1. **Financial/regulatory:** selling trading strategy content touches risk claims; keep disclaimers explicit and honest (AI reviewer checks "safe, honest intent").
2. **Backtest realism:** skill must instruct realistic fee/slippage modeling (fee 0.0004, slippage 0.0003) or buyers' results will be fantasy numbers.
3. **Live-trading safety:** skill must not encourage real-fund trading; scope strictly to design/backtest.
4. **Secrets/IP:** never bundle bot code requiring Binance keys into the skill; encode *knowledge*, not credentials.
5. **Support load:** paid skills → buyer-specific license ID; piracy via fingerprint dashboard; updates via new versions.

---

## 8. Action checklist (ordered)

- [ ] Choose final skill name (`btc-dual-ml-strategy`).
- [ ] Write `SKILL.md` (frontmatter + body) per §4.2.
- [ ] Write `references/strategy-rules.md` and `references/feature-labeling.md`.
- [ ] Local trigger tests + output-quality test; zip with clean structure.
- [ ] Verify no secrets/dangerous content (§4.4).
- [ ] Creator account + Creator Mode + Stripe payout setup.
- [ ] Submit via `/dashboard/submit`; price $29; tags; listing description.
- [ ] Pass automated + manual review (expect ~24–48h).
- [ ] Go live; add to GitHub/README + socials.
- [ ] Marketing push (post, UGC, community); monitor Creator Studio funnel (views → cart → sales).

---

## 9. Next immediate step

Draft and locally test the first version of `SKILL.md` (frontmatter + body) and the two reference files, then submit. The richest asset to sell is the owner’s hard-won operational playbook (adaptive gates, post-mortems, fix history) — that narrative should drive listing copy and marketing.
