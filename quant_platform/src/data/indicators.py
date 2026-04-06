from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

try:
    import pandas_ta as pta
except Exception:  # pragma: no cover
    pta = None

try:
    import ta
    from ta import momentum as ta_momentum
    from ta import trend as ta_trend
    from ta import volatility as ta_volatility
    from ta import volume as ta_volume
except Exception:  # pragma: no cover
    ta = None
    ta_momentum = None
    ta_trend = None
    ta_volatility = None
    ta_volume = None


TRADINGVIEW_MAP: Dict[str, str] = {
    "EMA": "ema_20",
    "SMA": "sma_20",
    "DEMA": "dema_20",
    "TEMA": "tema_20",
    "VWAP": "vwap",
    "Supertrend": "supertrend",
    "Ichimoku": "ichimoku_a",
    "RSI": "rsi_14",
    "RSI2": "rsi_2",
    "StochRSI": "stochrsi_k",
    "MACD": "macd",
    "MACD_H": "macdh",
    "ROC": "roc_10",
    "WilliamsR": "willr",
    "CCI": "cci",
    "UltimateOsc": "uo",
    "ATR": "atr_14",
    "BBWidth": "bb_width",
    "BBPctB": "bb_pctb",
    "Keltner": "kc_width",
    "ADX": "adx_14",
    "OBV": "obv",
    "MFI": "mfi",
    "HMA": "hma_20",
    "ALMA": "alma_20",
    "KAMA": "kama_10",
    "T3": "t3_10",
    "RMA": "rma_14",
    "SMMA": "smma_20",
    "Donchian": "donchian_mid_20",
    "Squeeze": "squeeze_on",
    "PSAR": "psar",
    "PPO": "ppo",
    "TRIX": "trix",
    "CMO": "cmo",
    "DPO": "dpo",
    "KST": "kst",
    "TSI": "tsi",
    "Vortex": "vortex_pos",
    "Chop": "chop",
}


def _ensure_series(value: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 1:
            return value.iloc[:, 0]
        return value.iloc[:, 0]
    return value


def _ema(series: pd.Series, length: int) -> pd.Series:
    if pta is not None:
        return pta.ema(series, length=length)
    if ta_trend is not None:
        return ta_trend.ema_indicator(series, window=length)
    return series.ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    if pta is not None:
        return pta.sma(series, length=length)
    if ta_trend is not None:
        return ta_trend.sma_indicator(series, window=length)
    return series.rolling(length).mean()


def _wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def _rma(series: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / length
    return series.ewm(alpha=alpha, adjust=False).mean()


def _smma(series: pd.Series, length: int) -> pd.Series:
    return _rma(series, length)


def _dema(series: pd.Series, length: int) -> pd.Series:
    ema1 = _ema(series, length)
    ema2 = _ema(ema1, length)
    return 2 * ema1 - ema2


def _tema(series: pd.Series, length: int) -> pd.Series:
    ema1 = _ema(series, length)
    ema2 = _ema(ema1, length)
    ema3 = _ema(ema2, length)
    return 3 * (ema1 - ema2) + ema3


def _alma(series: pd.Series, length: int, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    if length <= 1:
        return series
    m = offset * (length - 1)
    s = length / sigma
    weights = np.array([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(length)])
    weights /= weights.sum()
    return series.rolling(length).apply(lambda x: np.dot(x, weights), raw=True)


def _kama(series: pd.Series, length: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    if ta_trend is not None and hasattr(ta_trend, "kama"):
        return ta_trend.kama(series, window=length, pow1=fast, pow2=slow)
    change = series.diff(length).abs()
    volatility = series.diff().abs().rolling(length).sum()
    er = change / volatility.replace(0, np.nan)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = series.copy()
    for i in range(1, len(series)):
        if np.isnan(sc.iloc[i]):
            kama.iloc[i] = kama.iloc[i - 1]
        else:
            kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i - 1])
    return kama


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical = (high + low + close) / 3.0
    return (typical * volume).cumsum() / volume.replace(0, np.nan).cumsum()


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    if pta is not None:
        return pta.rsi(series, length=length)
    if ta_momentum is not None:
        return ta_momentum.rsi(series, window=length)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _rma(gain, length)
    avg_loss = _rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _stochrsi(series: pd.Series, length: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> tuple[pd.Series, pd.Series]:
    if pta is not None:
        out = pta.stochrsi(series, length=length)
        return out.iloc[:, 0], out.iloc[:, 1]
    rsi = _rsi(series, length)
    min_rsi = rsi.rolling(length).min()
    max_rsi = rsi.rolling(length).max()
    stoch = (rsi - min_rsi) / (max_rsi - min_rsi)
    k = stoch.rolling(smooth_k).mean() * 100
    d = k.rolling(smooth_d).mean()
    return k, d


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    if pta is not None:
        macd = pta.macd(series, fast=fast, slow=slow, signal=signal)
        return macd.iloc[:, 0], macd.iloc[:, 1], macd.iloc[:, 2]
    if ta_trend is not None:
        macd_ind = ta_trend.MACD(series, window_slow=slow, window_fast=fast, window_sign=signal)
        return macd_ind.macd(), macd_ind.macd_signal(), macd_ind.macd_diff()
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    if pta is not None:
        return pta.atr(high, low, close, length=length)
    if ta_volatility is not None:
        return ta_volatility.average_true_range(high, low, close, window=length)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return _rma(tr, length)


def _bollinger(close: pd.Series, length: int = 20, std_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if ta_volatility is not None:
        bb = ta_volatility.BollingerBands(close, window=length, window_dev=std_mult)
        mid = bb.bollinger_mavg()
        upper = bb.bollinger_hband()
        lower = bb.bollinger_lband()
    else:
        mid = _sma(close, length)
        std = close.rolling(length).std()
        upper = mid + std_mult * std
        lower = mid - std_mult * std
    width = (upper - lower) / mid.replace(0, np.nan)
    pctb = (close - lower) / (upper - lower).replace(0, np.nan)
    return upper, mid, lower, width, pctb


def _keltner(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20, mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if ta_volatility is not None:
        kc = ta_volatility.KeltnerChannel(high, low, close, window=length, window_atr=length)
        upper = kc.keltner_channel_hband()
        lower = kc.keltner_channel_lband()
        mid = kc.keltner_channel_mband()
    else:
        mid = _ema(close, length)
        atr = _atr(high, low, close, length)
        upper = mid + mult * atr
        lower = mid - mult * atr
    width = (upper - lower) / mid.replace(0, np.nan)
    return upper, mid, lower, width


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    if ta_momentum is not None and hasattr(ta_momentum, "williams_r"):
        return ta_momentum.williams_r(high, low, close, lbp=length)
    highest_high = high.rolling(length).max()
    lowest_low = low.rolling(length).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
    if ta_trend is not None and hasattr(ta_trend, "cci"):
        return ta_trend.cci(high, low, close, window=length)
    tp = (high + low + close) / 3.0
    sma = tp.rolling(length).mean()
    mad = tp.rolling(length).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def _ultimate_osc(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    if ta_momentum is not None and hasattr(ta_momentum, "ultimate_oscillator"):
        return ta_momentum.ultimate_oscillator(high, low, close)
    bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    tr = pd.concat([high, close.shift(1)], axis=1).max(axis=1) - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum().replace(0, np.nan)
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum().replace(0, np.nan)
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum().replace(0, np.nan)
    return 100 * (4 * avg7 + 2 * avg14 + avg28) / 7


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    if ta_volume is not None and hasattr(ta_volume, "on_balance_volume"):
        return ta_volume.on_balance_volume(close, volume)
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    if ta_volume is not None and hasattr(ta_volume, "money_flow_index"):
        return ta_volume.money_flow_index(high, low, close, volume, window=length)
    tp = (high + low + close) / 3.0
    raw = tp * volume
    pos = raw.where(tp > tp.shift(1), 0.0)
    neg = raw.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(length).sum()
    neg_sum = neg.rolling(length).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


def _ad_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return (clv * volume).cumsum()


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 20) -> pd.Series:
    if ta_volume is not None and hasattr(ta_volume, "chaikin_money_flow"):
        return ta_volume.chaikin_money_flow(high, low, close, volume, window=length)
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = clv * volume
    return mfv.rolling(length).sum() / volume.rolling(length).sum().replace(0, np.nan)


def _force_index(close: pd.Series, volume: pd.Series, length: int = 1) -> pd.Series:
    if ta_volume is not None and hasattr(ta_volume, "force_index"):
        return ta_volume.force_index(close, volume, window=length)
    return close.diff(length) * volume


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    if ta_trend is not None and hasattr(ta_trend, "ADXIndicator"):
        return ta_trend.ADXIndicator(high, low, close, window=length).adx()
    # Fallback: return NaN if unavailable
    return pd.Series(index=close.index, dtype=float)


def _psar(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    if ta_trend is not None and hasattr(ta_trend, "psar_up"):
        up = ta_trend.psar_up(high, low, close)
        down = ta_trend.psar_down(high, low, close)
        return up.fillna(down)
    return pd.Series(index=close.index, dtype=float)


def _ppo(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    return (ema_fast - ema_slow) / ema_slow.replace(0, np.nan) * 100


def _trix(close: pd.Series, length: int = 15) -> pd.Series:
    ema1 = _ema(close, length)
    ema2 = _ema(ema1, length)
    ema3 = _ema(ema2, length)
    return ema3.pct_change() * 100


def _cmo(close: pd.Series, length: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).rolling(length).sum()
    down = (-diff.clip(upper=0)).rolling(length).sum()
    return 100 * (up - down) / (up + down).replace(0, np.nan)


def _dpo(close: pd.Series, length: int = 20) -> pd.Series:
    shift = int(length / 2 + 1)
    return close.shift(shift) - _sma(close, length)


def _kst(close: pd.Series) -> pd.Series:
    roc1 = close.pct_change(10) * 100
    roc2 = close.pct_change(15) * 100
    roc3 = close.pct_change(20) * 100
    roc4 = close.pct_change(30) * 100
    return (
        roc1.rolling(10).mean()
        + 2 * roc2.rolling(10).mean()
        + 3 * roc3.rolling(10).mean()
        + 4 * roc4.rolling(15).mean()
    )


def _tsi(close: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
    diff = close.diff()
    abs_diff = diff.abs()
    ema1 = _ema(diff, r)
    ema2 = _ema(ema1, s)
    ema1_abs = _ema(abs_diff, r)
    ema2_abs = _ema(ema1_abs, s)
    return 100 * ema2 / ema2_abs.replace(0, np.nan)


def _vortex(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> tuple[pd.Series, pd.Series]:
    if ta_trend is not None and hasattr(ta_trend, "vortex_indicator_pos"):
        pos = ta_trend.vortex_indicator_pos(high, low, close, window=length)
        neg = ta_trend.vortex_indicator_neg(high, low, close, window=length)
        return pos, neg
    tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()
    pos = vm_plus.rolling(length).sum() / tr.rolling(length).sum().replace(0, np.nan)
    neg = vm_minus.rolling(length).sum() / tr.rolling(length).sum().replace(0, np.nan)
    return pos, neg


def _chop(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    sum_tr = tr.rolling(length).sum()
    highest = high.rolling(length).max()
    lowest = low.rolling(length).min()
    return 100 * np.log10(sum_tr / (highest - lowest).replace(0, np.nan)) / np.log10(length)


def _nvi_pvi(close: pd.Series, volume: pd.Series) -> tuple[pd.Series, pd.Series]:
    nvi = pd.Series(index=close.index, dtype=float)
    pvi = pd.Series(index=close.index, dtype=float)
    nvi.iloc[0] = 1000
    pvi.iloc[0] = 1000
    for i in range(1, len(close)):
        if volume.iloc[i] < volume.iloc[i - 1]:
            nvi.iloc[i] = nvi.iloc[i - 1] * (1 + close.pct_change().iloc[i])
        else:
            nvi.iloc[i] = nvi.iloc[i - 1]
        if volume.iloc[i] > volume.iloc[i - 1]:
            pvi.iloc[i] = pvi.iloc[i - 1] * (1 + close.pct_change().iloc[i])
        else:
            pvi.iloc[i] = pvi.iloc[i - 1]
    return nvi, pvi


def _hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    series = series.dropna()
    if len(series) < max_lag + 2:
        return np.nan
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(series[lag:] - series[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


def compute_indicators(df: pd.DataFrame, mode: str = "expanded") -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    close = _ensure_series(out["close"])
    high = _ensure_series(out["high"])
    low = _ensure_series(out["low"])
    open_ = _ensure_series(out["open"])
    volume = _ensure_series(out["volume"])

    # Core set
    out["ema_9"] = _ema(close, 9)
    out["ema_21"] = _ema(close, 21)
    out["ema_50"] = _ema(close, 50)
    out["ema_200"] = _ema(close, 200)
    out["sma_20"] = _sma(close, 20)
    out["dema_20"] = _dema(close, 20)
    out["tema_20"] = _tema(close, 20)
    out["vwap"] = _vwap(high, low, close, volume)

    if pta is not None:
        supertrend = pta.supertrend(high, low, close)
        if supertrend is not None and "SUPERT_7_3.0" in supertrend.columns:
            out["supertrend"] = supertrend["SUPERT_7_3.0"]
        ichimoku = pta.ichimoku(high, low)
        if ichimoku is not None:
            out["ichimoku_a"] = ichimoku[0]["ISA_9"]
            out["ichimoku_b"] = ichimoku[0]["ISB_26"]
    elif ta_trend is not None and hasattr(ta_trend, "IchimokuIndicator"):
        ichi = ta_trend.IchimokuIndicator(high, low)
        out["ichimoku_a"] = ichi.ichimoku_a()
        out["ichimoku_b"] = ichi.ichimoku_b()
    else:
        out["ichimoku_a"] = np.nan
        out["ichimoku_b"] = np.nan

    out["rsi_14"] = _rsi(close, 14)
    out["rsi_2"] = _rsi(close, 2)
    stoch_k, stoch_d = _stochrsi(close)
    out["stochrsi_k"] = stoch_k
    out["stochrsi_d"] = stoch_d
    macd_line, macd_signal, macd_hist = _macd(close)
    out["macd"] = macd_line
    out["macdh"] = macd_hist
    out["macds"] = macd_signal
    out["roc_10"] = close.pct_change(10) * 100
    out["willr"] = _williams_r(high, low, close)
    out["cci"] = _cci(high, low, close)
    out["uo"] = _ultimate_osc(high, low, close)

    out["atr_14"] = _atr(high, low, close, 14)
    bb_upper, bb_mid, bb_lower, bb_width, bb_pctb = _bollinger(close)
    out["bb_width"] = bb_width
    out["bb_pctb"] = bb_pctb
    kc_upper, kc_mid, kc_lower, kc_width = _keltner(high, low, close)
    out["kc_width"] = kc_width

    out["obv"] = _obv(close, volume)
    out["mfi"] = _mfi(high, low, close, volume)
    out["ad"] = _ad_line(high, low, close, volume)
    out["cmf_20"] = _cmf(high, low, close, volume)
    out["force_index"] = _force_index(close, volume)

    out["adx_14"] = _adx(high, low, close)

    out["hist_vol_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    out["real_vol_5"] = close.pct_change().rolling(5).std() * np.sqrt(252)
    out["real_vol_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    out["pivot_point"] = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
    rolling_high = close.rolling(252).max()
    rolling_low = close.rolling(252).min()
    out["dist_52w_high"] = (close - rolling_high) / rolling_high.replace(0, np.nan)
    out["dist_52w_low"] = (close - rolling_low) / rolling_low.replace(0, np.nan)
    out["fib_38"] = rolling_low + 0.382 * (rolling_high - rolling_low)
    out["fib_62"] = rolling_low + 0.618 * (rolling_high - rolling_low)

    out["hurst_100"] = close.rolling(100).apply(_hurst_exponent, raw=False)

    if mode == "core":
        return out

    # Expanded set
    out["wma_20"] = _wma(close, 20)
    out["hma_20"] = _wma(2 * _wma(close, 10) - _wma(close, 20), int(np.sqrt(20)))
    out["alma_20"] = _alma(close, 20)
    out["kama_10"] = _kama(close, 10)
    out["t3_10"] = _tema(close, 10)  # approximate if pandas_ta unavailable
    out["rma_14"] = _rma(close, 14)
    out["smma_20"] = _smma(close, 20)

    out["psar"] = _psar(high, low, close)
    out["ppo"] = _ppo(close)
    out["trix"] = _trix(close)
    out["cmo"] = _cmo(close)
    out["dpo"] = _dpo(close)
    out["kst"] = _kst(close)
    out["tsi"] = _tsi(close)
    vortex_pos, vortex_neg = _vortex(high, low, close)
    out["vortex_pos"] = vortex_pos
    out["vortex_neg"] = vortex_neg
    out["chop"] = _chop(high, low, close)
    out["bop"] = (close - open_) / (high - low).replace(0, np.nan)

    nvi, pvi = _nvi_pvi(close, volume)
    out["nvi"] = nvi
    out["pvi"] = pvi

    dc_high = high.rolling(20).max()
    dc_low = low.rolling(20).min()
    dc_mid = (dc_high + dc_low) / 2
    out["donchian_high_20"] = dc_high
    out["donchian_low_20"] = dc_low
    out["donchian_mid_20"] = dc_mid
    out["donchian_width_20"] = (dc_high - dc_low) / dc_mid.replace(0, np.nan)

    if "bb_width" in out.columns and "kc_width" in out.columns:
        out["squeeze_on"] = (out["bb_width"] < out["kc_width"]).astype(float)

    return out
