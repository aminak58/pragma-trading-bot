import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure src on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from regime.hmm_detector import RegimeDetector


def winsorize(series: pd.Series, p_low: float = 0.01, p_high: float = 0.99) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return series
    lo = s.quantile(p_low)
    hi = s.quantile(p_high)
    return series.clip(lower=lo, upper=hi)


def rolling_z(series: pd.Series, window: int = 300) -> pd.Series:
    m = series.rolling(window).mean()
    sd = series.rolling(window).std()
    return (series - m) / (sd + 1e-9)


def label_with_hysteresis(score: pd.Series,
                          p_up_in=0.80, p_up_out=0.70,
                          p_dn_in=0.20, p_dn_out=0.30,
                          p_up_late_in=0.90, p_up_late_out=0.85,
                          p_dn_late_in=0.10, p_dn_late_out=0.15,
                          median_window: int = 5) -> pd.Series:
    s = score.dropna()
    if s.empty:
        return pd.Series(index=score.index, dtype='object')

    # Percentiles on this sample (post alignment)
    up_in = s.quantile(p_up_in)
    up_out = s.quantile(p_up_out)
    dn_in = s.quantile(p_dn_in)
    dn_out = s.quantile(p_dn_out)
    up_late_in = s.quantile(p_up_late_in)
    up_late_out = s.quantile(p_up_late_out)
    dn_late_in = s.quantile(p_dn_late_in)
    dn_late_out = s.quantile(p_dn_late_out)

    labels = pd.Series('Neutral', index=score.index, dtype='object')

    state = 'Neutral'
    for t, v in score.items():
        if pd.isna(v):
            labels.at[t] = state
            continue
        if state == 'Neutral':
            if v >= up_in:
                state = 'Up_Early'
            elif v <= dn_in:
                state = 'Down_Early'
            elif v >= up_late_in:
                state = 'Up_Late'
            elif v <= dn_late_in:
                state = 'Down_Late'
        elif state == 'Up_Early':
            if v < up_out:
                state = 'Neutral'
        elif state == 'Down_Early':
            if v > dn_out:
                state = 'Neutral'
        elif state == 'Up_Late':
            if v < up_late_out:
                state = 'Neutral'
        elif state == 'Down_Late':
            if v > dn_late_out:
                state = 'Neutral'
        labels.at[t] = state

    # Median filter to reduce flip-flops
    if median_window and median_window > 1:
        as_int = labels.map({'Down_Late': -2, 'Down_Early': -1, 'Neutral': 0, 'Up_Early': 1, 'Up_Late': 2}).fillna(0)
        filt = as_int.rolling(median_window, center=True, min_periods=1).median().round().astype(int)
        labels = filt.map({-2: 'Down_Late', -1: 'Down_Early', 0: 'Neutral', 1: 'Up_Early', 2: 'Up_Late'})

    return labels


def label_simple(score: pd.Series) -> pd.Series:
    s = score.dropna()
    if s.empty:
        return pd.Series(index=score.index, dtype='object')
    p25 = s.quantile(0.25)
    p40 = s.quantile(0.40)
    p60 = s.quantile(0.60)
    p75 = s.quantile(0.75)
    labels = pd.Series('Neutral', index=score.index, dtype='object')
    labels[score >= p75] = 'Up_Early'
    labels[(score >= p60) & (score < p75)] = 'Up_Late'
    labels[score <= p25] = 'Down_Early'
    labels[(score > p25) & (score < p40)] = 'Down_Late'
    return labels


def summarize_stability(labels: pd.Series) -> tuple:
    if labels.empty:
        return (0, float('nan'), float('nan'), {})
    runs = (labels != labels.shift(1)).cumsum()
    dwell = labels.groupby(runs).size().median() if labels.notna().any() else np.nan
    flip = float((labels != labels.shift(1)).mean() * 100) if len(labels) else np.nan
    coverage = labels.value_counts(normalize=True).to_dict()
    coverage = {k: round(v * 100, 2) for k, v in coverage.items()}
    return (len(labels), dwell, flip, coverage)


def analyze_file(path: Path, timeframe_label: str):
    print(f'FILE {path.name} TF {timeframe_label}')
    if not path.exists():
        print('MISSING')
        return

    df = pd.read_feather(path)
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)

    # Build minimal frame
    base = pd.DataFrame(index=df.index)
    base['close'] = df['close']

    rd = RegimeDetector()

    # 1) Score raw -> winsorize -> rolling z-score
    raw_score = rd.compute_trend_phase_score(df)
    score_w = winsorize(raw_score, 0.01, 0.99)
    score = rolling_z(score_w, window=300).rename('score')

    # 2) Forward returns (no leakage) and trim tail H bars
    fret5 = np.log(base['close'].shift(-5) / base['close']).rename('f5')
    fret20 = np.log(base['close'].shift(-20) / base['close']).rename('f20')
    valid = base.index[:-20]
    score = score.loc[valid]
    fret5 = fret5.loc[valid]
    fret20 = fret20.loc[valid]
    fret5 = winsorize(fret5, 0.01, 0.99)
    fret20 = winsorize(fret20, 0.01, 0.99)

    # Hysteresis labels (A) and simple labels (B)
    labels_A = label_with_hysteresis(score)
    labels_B = label_simple(score)

    # Train HMM and regimes/conf aligned
    try:
        look = min(len(df), 1500)
        rd.train(df, lookback=look)
        df_valid = df.loc[valid]
        regimes, conf = rd.predict_regime_sequence(df_valid)
        n = len(regimes)
        idx_tail = valid[-n:]
        reg_s = pd.Series(regimes, index=idx_tail, dtype='object').rename('regime')
        conf_s = pd.Series(conf, index=idx_tail, dtype=float).rename('conf')
        common_idx = score.index.intersection(fret5.index).intersection(fret20.index).intersection(reg_s.index)
        score = score.loc[common_idx]
        fret5 = fret5.loc[common_idx]
        fret20 = fret20.loc[common_idx]
        labels_A = labels_A.loc[common_idx]
        labels_B = labels_B.loc[common_idx]
        reg_s = reg_s.loc[common_idx]
        conf_s = conf_s.loc[common_idx]
    except Exception as e:
        print('HMM_ERROR', str(e))
        common_idx = score.index
        reg_s = pd.Series(index=common_idx, dtype='object', data=np.nan, name='regime')
        conf_s = pd.Series(index=common_idx, dtype=float, data=np.nan, name='conf')

    # Build aligned frames A (gated) and B (no gating)
    aligned_A = pd.concat([score, fret5, fret20, labels_A.rename('label'), reg_s, conf_s], axis=1).dropna()
    aligned_B = pd.concat([score, fret5, fret20, labels_B.rename('label'), reg_s, conf_s], axis=1).dropna()

    gated_A = aligned_A[aligned_A['conf'] >= 0.6] if 'conf' in aligned_A.columns else aligned_A

    # Stability summaries
    COUNT_A, DWELL_A, FLIP_A, COVER_A = summarize_stability(aligned_A['label'])
    COUNT_B, DWELL_B, FLIP_B, COVER_B = summarize_stability(aligned_B['label'])

    # Conditional correlations helper
    def corr_pair(df_in: pd.DataFrame) -> dict:
        res = {}
        for reg in ['trending', 'low_volatility', 'high_volatility']:
            sub = df_in[df_in['regime'] == reg]
            if len(sub) >= 30:
                res[reg] = {
                    'corr5': float(sub['score'].corr(sub['f5'])),
                    'corr20': float(sub['score'].corr(sub['f20'])),
                    'count': int(len(sub))
                }
            else:
                res[reg] = {'corr5': np.nan, 'corr20': np.nan, 'count': int(len(sub))}
        if len(df_in) >= 30:
            res['ALL'] = {
                'corr5': float(df_in['score'].corr(df_in['f5'])),
                'corr20': float(df_in['score'].corr(df_in['f20'])),
                'count': int(len(df_in))
            }
        else:
            res['ALL'] = {'corr5': np.nan, 'corr20': np.nan, 'count': int(len(df_in))}
        return res

    CORR_A = corr_pair(gated_A)
    CORR_B = corr_pair(aligned_B)

    # Hourly session effect (A scenario)
    if 'date' in df.columns and len(aligned_A):
        hours = pd.to_datetime(df.loc[aligned_A.index, 'date']).dt.hour
        hour_mean_A = aligned_A.groupby(hours).agg({'score': 'mean'})['score'].to_dict()
        hour_mean_A = {int(k): round(float(v), 4) for k, v in hour_mean_A.items()}
    else:
        hour_mean_A = {}

    print('SCENARIO A (hysteresis+gating)')
    print('STABILITY', {'COUNT': COUNT_A, 'dwell': DWELL_A, 'flip': FLIP_A, 'coverage': COVER_A})
    print('COND_CORR', CORR_A)
    print('HOUR_MEAN_SCORE', hour_mean_A)

    print('SCENARIO B (no hysteresis/gating)')
    print('STABILITY', {'COUNT': COUNT_B, 'dwell': DWELL_B, 'flip': FLIP_B, 'coverage': COVER_B})
    print('COND_CORR', CORR_B)


def main():
    base = ROOT / 'user_data' / 'data' / 'binance'
    targets = [
        (base / 'BTC_USDT-5m.feather', '5m'),
        (base / 'ETH_USDT-5m.feather', '5m'),
        (base / 'BTC_USDT-1h.feather', '1h'),
    ]
    for p, tf in targets:
        analyze_file(p, tf)


if __name__ == '__main__':
    main()
