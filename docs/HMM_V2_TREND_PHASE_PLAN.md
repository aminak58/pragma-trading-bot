## HMM v2 - Trend Phase Score and 5-State Roadmap

### Goal
Stabilize directional understanding via a composite Trend Phase Score on current 3-state HMM, validate separability of Early/Late phases, then upgrade to 5-state with robust mapping and hysteresis.

### Step 1 — Trend Phase Score on 3-state
- Directional features (no n_states change yet):
  - Slopes/acceleration: slopes of EMA20/50/100, second differences, MACD slope
  - Normalized momentum: z-score of price vs EMA, distance to BB mid, ROC
  - Phase location: distance to rolling high/low, recovery-from-low percent
  - Trend quality: ADX, BB width, ATR percent
  - Exhaustion: RSI percentile, returns skewness
- Build composite score (weighted sum of selected standardized features)
- Post-label analysis: Check if score distributions separate Early/Late within current 3 regimes; retune weights if not

### Step 2 — Upgrade to 5-state with auto-labeling
- n_states=5 with mapping rules:
  - Uptrend_Early: strong positive slope, increasing distance above EMA, low distance to rolling high
  - Uptrend_Late: still positive slope but slowing, high z-score/RSI, large distance to high (exhaustion)
  - Downtrend_Early: mirror of Up_Early with fresh negatives
  - Downtrend_Late: weakening negatives, oversold, large distance from rolling low
  - Sideways: low ADX, small BB width, near-zero slopes
- Add hysteresis: min dwell time, smoothing window, penalty on flip-flops

### Step 3 — Validation
- Rolling/walk-forward splits
- Transition matrix sanity: Early→Late common, Sideways connects, Early→opposite Late rare
- Consistency score: mapping rules vs HMM output agreement

### Step 4 — Strategy Integration (Freqtrade)
- Decision mapping and confidence thresholds per phase
- Min candles since regime change before entries

### Step 5 — Acceptance
- Higher trade rate with similar MDD, improved Sharpe/Calmar, stable PnL distribution

### Deliverables
- compute_trend_phase_score() method scaffold
- 3-state validation notebook/script (later)
- 5-state mapping in detector with hysteresis (later)


