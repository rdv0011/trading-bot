# Fix "Way Too Many Requests" — Binance API Rate Limiting

## Incident Summary

Between 2026-06-19 23:55 and 2026-06-20 04:35 the bot triggered **Binance `-1003` rate limit bans** across **6 different outbound IPs** (15.158.219.105, .106, .71, .69, .66, .74). The bans lasted anywhere from ~10 minutes to over 2 hours. During ban windows the bot was unable to:
- Fetch account balance (`futures_account_balance`) — caused full iteration crashes
- Fetch historical klines (`futures_klines`) — blocked tactical/strategic predictions
- Fetch positions / leverage (`futures_position_information`) — degraded state tracking

The bot never recovered from a ban during the observed window — each iteration retried the same calls and got re-banned.

---

## Root-Cause Analysis

**6 distinct problems** were identified. All contribute to exceeding Binance's 1200-weight-per-minute limit for the Futures API.

---

### Problem 1: No client-side weight-based rate limiter

**Severity: Critical**

**Evidence**: Every `Client(...)` instantiation lacks rate limiting configuration:

- `binancefuturesbroker.py:13` — `Client(api_key=..., api_secret=..., testnet=...)`
- `binancespotbroker.py:31` — `Client(**kwargs)`
- `mltraining.py:617` — `Client(api_key, api_secret, testnet=True)`
- `dualmlsimulation.py:158` — `Client()`
- `strategictraining.py:224` — `Client()`

No file imports or configures `requests_params` with weight-based rate limiting. The python-binance client supports `requests_params={"timeout": 20}` but does **not** enforce Binance's per-minute weight quota out of the box. When the python-binance client lacks explicit rate limiting, every `call` API call consumes weight without tracking cumulative usage.

**Fix**: Wrap the client with a weight-aware rate limiter (e.g. `BinanceRateLimiter` from `binance.helpers` or a custom token-bucket that tracks Binance response headers `x-mbx-used-weight` and `x-mbx-used-weight-1m`). Reject calls before they're sent if the weight budget is exhausted.

---

### Problem 2: No TTL cache for historical klines — 3 full-history fetches per cycle

**Severity: Critical**

**Evidence**: `on_trading_iteration()` in `dualmlstrategy.py` calls `get_historical_prices()` **3 times** per 5-minute cycle:

| Line | Call | Candles fetched |
|------|------|----------------|
| 105 | `df_tactical = self.get_historical_prices(asset, 600, "5m")` | ~600 candles (50h) |
| 114 | `df_strategic = self.get_historical_prices(asset, 300, "1h")` | ~300 candles (300h) |
| 127 | `df_raw_tactical = self.get_historical_prices(asset, 600, "5m")` | ~600 candles (50h) **duplicate** |

That's potentially **1,500 candles fetched per cycle**, most of which are identical to the previous cycle's data (only 1–2 new candles since last poll).

**Estimated weight per cycle** (Binance Futures API weights):
- `GET /futures/klines` (1h, limit=300): weight ≈ 33 (adjusted for limit/100)
- `GET /futures/klines` (5m, limit=600): weight ≈ 44 (adjusted for limit/100)
- Each call ≈ 30–45 weight units
- 3 calls × ~40 = **~120 weight per cycle**
- Over 60s window: if all 3 fire within the same second, that's ~120 weight consumed instantly

**Fix**: Add an in-memory TTL cache keyed by `(symbol, timeframe)` with a cache duration of `timeframe_minutes * 0.8` (e.g., 4 minutes for 5m data). Only fetch from Binance when:
- The cache is empty (first call after startup), OR
- The cached data is older than the TTL, OR
- The cache has fewer candles than requested (shouldn't happen normally)

---

### Problem 3: Balance cache declared but never wired

**Severity: High**

**Evidence**: `BinanceBaseBroker.__init__()` (binancebasebroker.py:47-49) initializes cache fields:

```python
self._cached_balance = None
self._balance_cache_time = 0
self._balance_cache_duration = 5
```

But neither `BinanceFuturesBroker.get_cash()` (binancefuturesbroker.py:20) nor `BinanceSpotBroker.get_cash()` (binancespotbroker.py:34) checks these fields before calling the API:

```python
def get_cash(self, quote_asset_symbol="USDT") -> float:
    balances = self.client.futures_account_balance()  # ← always hits API
    bal = next(...)
    return float(bal["balance"]) if bal else 0.0
```

`futures_account_balance()` has a Binance weight of **25**. Combined with the other position info calls, this adds significant weight. The `get_cash()` is the **first call** in every iteration (dualmlstrategy.py:87) — if it fails, the entire iteration crashes.

**Fix**: Wire the existing cache fields:
- On `get_cash()`, if `time.time() - _balance_cache_time < _balance_cache_duration`, return `_cached_balance`
- Otherwise fetch, cache, and return
- Invalidate cache after any trade execution

---

### Problem 4: Redundant per-iteration polling of static exchange state

**Severity: High**

**Evidence**: Every 5-minute iteration, `on_trading_iteration()` unconditionally fetches:

| Line | Call | Endpoint | Binance Weight |
|------|------|----------|----------------|
| 87 | `get_cash()` → `futures_account_balance()` | `GET /fapi/v2/balance` | 25 |
| 94 | `get_position(asset)` → `futures_position_information()` | `GET /fapi/v2/positionRisk` | 10 |
| 95 | `get_position_leverage(asset)` → `futures_position_information()` | `GET /fapi/v2/positionRisk` | 10 |

During the ~4.5-hour observed log, the bot was **FLAT** the entire time. Leverage varied only between 4.2x–4.7x (minor changes from strategic model prediction, not from exchange state). Position was always `None`/`FLAT`.

Despite this, these three calls were made **every 5 minutes** without any check on whether state had actually changed.

**Fix**:
- **Leverage**: Cache the last-known leverage value. Only re-fetch from exchange when opening a new position (PositionManager already calls `set_leverage()` before entry).
- **Position**: The `DualMLStrategy.get_position()` call at line 94 is partially redundant with PositionManager's internal state tracking. When `_state is None`, the position is flat — skip the API call and log "FLAT" from cached knowledge.
- **Balance**: After caching as described in Problem 3, only fetch at most once every 5 seconds (the `_balance_cache_duration` already specifies this, it just isn't used).

---

### Problem 5: Duplicate `df_raw_tactical` historical fetch

**Severity: Medium**

**Evidence**: In `dualmlstrategy.py:on_trading_iteration()`:

```python
# Line 105: Fetch for feature/label computation
df_tactical = self.get_historical_prices(asset, 600, "5m")

# ... lines 123-125: make_features + make_labels + get_features ...

# Line 127: Fetch SAME data again for prediction
df_raw_tactical = self.get_historical_prices(asset, 600, "5m")
```

Both calls fetch **identical** data (same symbol, same length `600`, same timeframe `"5m"`). The second result `df_raw_tactical` is used only for `make_features(df_raw_tactical, ...).iloc[[-1]]` — i.e., just the last row to make the next prediction.

**Fix**: Reuse `df_tactical` instead of re-fetching. Take the last row after feature engineering:

```python
df_pred = df_tactical.iloc[[-1]]       # before make_features → error (already featured)
# OR better:
df_pred = df_tactical.tail(1).copy()    # after features are computed
```

This eliminates 1 of the 3 historical data fetches, reducing klines weight by ~40 per cycle.

---

### Problem 6: No rate-limit backoff or circuit breaker

**Severity: High**

**Evidence**: When a `-1003` error is received:

- `get_historical_prices()` in `binancebasebroker.py:226` logs the error and returns `None` — no sleep, no backoff
- `get_position()` in `binancefuturesbroker.py:36` logs and returns `None` — no sleep, no backoff
- `get_position_leverage()` in `binancefuturesbroker.py:164` logs and returns `None` — no sleep, no backoff
- `get_cash()` **does not catch exceptions at all** — it propagates up to `basestrategy.py:108` which catches the generic exception, sleeps **60 seconds** (line 110), then retries the full iteration including the same failing calls.

The 60-second sleep in the iteration-level catch is too short for Binance bans that last **10+ minutes**, and it only applies after a full crash path — individual per-call failures (historical prices, position, leverage) sleep 0 seconds.

Log evidence:
- 00:05:07 — First `-1003` (fetch leverage) — bot keeps running
- 00:05:27 — Tactical & Strategic still execute (presumably using stale data)
- 00:10:25 — Normal operation briefly recovers
- 00:15:06 — `-1003` (historical prices) — "Insufficient tactical data, skipping"
- Iterations continue failing with `-1003` for hours, always retrying at full throttle

**Fix**: Implement a multi-level backoff:
1. **Per-call level**: On `-1003`, apply exponential backoff (start 30s, max 300s) *for that specific call*
2. **Per-endpoint circuit breaker**: Track consecutive failures per endpoint category; after N failures, suspend all calls to that category for `backoff * (2^failures)` seconds
3. **Global kill switch**: If any `-1003` is received, reduce global request rate by 50% for 5 minutes
4. **Iteration-level**: Increase the generic catch sleep from 60s to `min(300, 60 * 2^consecutive_failures)`
5. **Ban timestamp tracking**: Parse the `banned until <epoch_ms>` from the error message and skip that endpoint until the ban expires

---

## Summary Table

| # | Problem | Root File(s) | Severity | Fix Type |
|---|---------|-------------|----------|----------|
| 1 | No weight-based rate limiter on Client | `binancefuturesbroker.py`, `binancespotbroker.py`, `mltraining.py`, `dualmlsimulation.py`, `strategictraining.py` | Critical | Add token-bucket / `BinanceRateLimiter` wrapper |
| 2 | No TTL cache for historical klines | `binancebasebroker.py` (`get_historical_prices`), `dualmlstrategy.py` (3 calls/cycle) | Critical | Add `(symbol, timeframe)` → `(df, timestamp)` cache |
| 3 | Balance cache declared but unused | `binancefuturesbroker.py` (`get_cash`) | High | Wire `_cached_balance` / `_balance_cache_time` |
| 4 | Redundant per-iteration polling of static state | `dualmlstrategy.py:87,94,95` | High | Cache leverage; skip position API when state is FLAT; leverage balance cache |
| 5 | Duplicate `df_raw_tactical` historical fetch | `dualmlstrategy.py:127` (vs line 105) | Medium | Reuse `df_tactical` instead of re-fetching |
| 6 | No rate-limit backoff or circuit breaker | `basestrategy.py:110`, `binancebasebroker.py:226`, `binancefuturesbroker.py:36,164` | High | Exponential backoff + circuit breaker on `-1003` |

## Recommended Order of Implementation

1. **Problem 2** (TTL cache for klines) — Highest impact on weight reduction, simplest change
2. **Problem 5** (duplicate df_raw_tactical) — Trivial dedup, eliminates 1 of 3 klines calls
3. **Problem 3** (balance cache) — Cache is already declared, just needs wiring
4. **Problem 4** (redundant polling) — Reduces position + leverage calls per cycle
5. **Problem 1** (rate limiter) — Safety net, prevents any single-cycle burst from triggering bans
6. **Problem 6** (backoff) — Last resort defense when bans do happen

Problems 2 + 5 combined should reduce klines API weight by **~66%**. Adding Problem 3 + 4 reduces balance/position/leverage weight by **~90%** during flat periods.
