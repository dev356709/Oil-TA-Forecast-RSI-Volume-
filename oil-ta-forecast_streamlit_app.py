# file: oil-ta-forecast_streamlit_app.py
"""
Oil Technical Forecast Dashboard (Streamlit) — Clean Rebuild

Features
- Oil only (WTI futures: CL=F)
- TA forecast for next N trading days using RSI + Volume + Drift; High/Low via ATR
- Intraday 15‑minute chart with RSI + Volume and auto notes
- Backtest last N days (1‑step ahead): MAE, MAPE, Coverage, Bias, StdDev + charts
- Auto‑tune (grid search) for TA params; apply best combo

Run
    pip install -U streamlit yfinance pandas numpy plotly pytz scipy
    streamlit run oil-ta-forecast_streamlit_app.py

Notes
- Data via Yahoo! Finance (yfinance). Educational only. Not financial advice.
- 15m data availability depends on Yahoo limits (try ≤30 days).
- Times shown in Europe/London.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timezone
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
import streamlit as st

try:
    from scipy import stats as _stats  # optional for QQ plot
except Exception:  # pragma: no cover
    _stats = None

import yfinance as yf


LONDON_TZ = pytz.timezone("Europe/London")
OIL_TICKER = "CL=F"


def to_london(ts: pd.DatetimeIndex | pd.Timestamp) -> pd.DatetimeIndex | pd.Timestamp:
    if isinstance(ts, pd.DatetimeIndex):
        if ts.tz is None:
            return ts.tz_localize("UTC").tz_convert(LONDON_TZ)
        return ts.tz_convert(LONDON_TZ)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc).astimezone(LONDON_TZ)
    return ts.astimezone(LONDON_TZ)


@st.cache_data(ttl=3600, show_spinner=False)
def load_yf(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV with robust column normalization."""
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",  # avoid MultiIndex for single ticker
    )
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(t[0]) if len(t) > 0 else "" for t in df.columns]
    df.columns = [c[0] if isinstance(c, tuple) and len(c) > 0 else c for c in df.columns]


    rename_map = {}
    for c in list(df.columns):
        cu = str(c).strip().lower()
        if cu.startswith("open"):
            rename_map[c] = "Open"
        elif cu.startswith("high"):
            rename_map[c] = "High"
        elif cu.startswith("low"):
            rename_map[c] = "Low"
        elif cu.startswith("close"):
            rename_map[c] = "Close"
        elif cu.startswith("adj close"):
            rename_map[c] = "Close"
        elif cu.startswith("volume"):
            rename_map[c] = "Volume"
    df = df.rename(columns=rename_map)

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep]

    df = df.sort_index()
    df.index = to_london(df.index)
    return df


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def volume_zscore(volume: pd.Series, lookback: int = 20) -> pd.Series:
    mean = volume.rolling(lookback, min_periods=max(2, lookback // 2)).mean()
    std = volume.rolling(lookback, min_periods=max(2, lookback // 2)).std()
    z = (volume - mean) / (std.replace(0, np.nan))
    return z.fillna(0.0).clip(-5, 5)


def business_days(start: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    out = []
    d = start
    while len(out) < n:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out


@dataclass
class ForecastParams:
    rsi_period: int = 14
    atr_period: int = 14
    atr_mult: float = 0.75
    ema_ret_span: int = 20
    vol_z_lookback: int = 20
    w_drift: float = 0.35
    w_meanrev: float = 0.50
    w_volmom: float = 0.15


@dataclass
class ForecastPoint:
    date: pd.Timestamp
    pred_open: float
    pred_high: float
    pred_low: float
    pred_close: float
    note: str


def one_step_ta_predict(last_close: float, last_rsi: float, last_volz: float, last_ema_ret: float,
                        last_atr: float, params: ForecastParams) -> tuple[float, float, float, float, str]:
    meanrev = -(last_rsi - 50) / 100.0
    trend_sign = np.sign(last_ema_ret) if not np.isnan(last_ema_ret) else 0.0
    volmom = trend_sign * (abs(last_volz) / 10.0)
    pred_ret = (
        params.w_drift * (last_ema_ret if not np.isnan(last_ema_ret) else 0.0)
        + params.w_meanrev * meanrev
        + params.w_volmom * volmom
    )
    pred_ret = float(np.clip(pred_ret, -0.05, 0.05))
    pred_close = last_close * (1.0 + pred_ret)

    rsi_extreme = max(0.0, (abs(last_rsi - 50) - 20) / 30)
    band = params.atr_mult * last_atr * (1.0 + 0.25 * rsi_extreme)
    pred_high = pred_close + band
    pred_low = max(0.01, pred_close - band)
    pred_open = (last_close + pred_close) / 2.0
    note = f"ret={pred_ret:+.2%}, rsi={last_rsi:.1f}, volZ={last_volz:.2f}, drift={last_ema_ret:+.2%}"
    return pred_open, pred_high, pred_low, pred_close, note


def ta_forecast_next_days(daily: pd.DataFrame, horizon: int, params: ForecastParams) -> List[ForecastPoint]:
    df = daily.copy()
    df["RSI"] = rsi(df["Close"], params.rsi_period)
    df["ATR"] = atr(df, params.atr_period)
    df["ret"] = df["Close"].pct_change()
    df["ema_ret"] = df["ret"].ewm(span=params.ema_ret_span, adjust=False).mean()
    df["vol_z"] = volume_zscore(df["Volume"], params.vol_z_lookback)
    df = df.dropna().copy()
    if df.empty:
        return []

    last = df.iloc[-1]
    prev_close = float(last["Close"])
    ema_ret = float(last["ema_ret"]) if not math.isnan(last["ema_ret"]) else 0.0
    rsi_val = float(last["RSI"]) if not math.isnan(last["RSI"]) else 50.0
    volz = float(last["vol_z"]) if not math.isnan(last["vol_z"]) else 0.0
    atr_val = float(last["ATR"]) if not math.isnan(last["ATR"]) else df["ATR"].dropna().iloc[-1]

    future_days = business_days(df.index[-1], horizon)
    out: List[ForecastPoint] = []

    for d in future_days:
        p_open, p_high, p_low, p_close, note = one_step_ta_predict(
            prev_close, rsi_val, volz, ema_ret, atr_val, params
        )
        out.append(ForecastPoint(d, p_open, p_high, p_low, p_close, note))
        # roll
        prev_close = p_close
        ema_ret = 0.9 * ema_ret + 0.1 * ((p_close / p_open) - 1.0)
        rsi_val = float(np.clip(50 + 0.9 * (rsi_val - 50) - 10 * ((p_close / p_open) - 1.0), 5, 95))
        volz *= 0.8
        atr_val *= 0.99

    return out


def backtest_last_n_days(daily: pd.DataFrame, n_days: int, params: ForecastParams) -> pd.DataFrame:
    df = daily.copy()
    df["RSI"] = rsi(df["Close"], params.rsi_period)
    df["ATR"] = atr(df, params.atr_period)
    df["ret"] = df["Close"].pct_change()
    df["ema_ret"] = df["ret"].ewm(span=params.ema_ret_span, adjust=False).mean()
    df["vol_z"] = volume_zscore(df["Volume"], params.vol_z_lookback)

    df = df.dropna().copy()
    if len(df) < n_days + 5:
        n_days = max(1, len(df) - 5)

    rows = []
    idxs = df.index
    for i in range(len(df) - n_days, len(df)):
        t = idxs[i]
        t_prev = idxs[i - 1]
        prev_row = df.loc[t_prev]
        last_close = float(prev_row["Close"])  # info up to t-1
        last_rsi = float(prev_row["RSI"]) if not math.isnan(prev_row["RSI"]) else 50.0
        last_volz = float(prev_row["vol_z"]) if not math.isnan(prev_row["vol_z"]) else 0.0
        last_ema_ret = float(prev_row["ema_ret"]) if not math.isnan(prev_row["ema_ret"]) else 0.0
        last_atr = float(prev_row["ATR"]) if not math.isnan(prev_row["ATR"]) else df["ATR"].dropna().iloc[max(0, i-2)]

        p_open, p_high, p_low, p_close, note = one_step_ta_predict(
            last_close, last_rsi, last_volz, last_ema_ret, last_atr, params
        )

        actual_close = float(df.loc[t, "Close"]) if t in df.index else np.nan
        rows.append({
            "Date": t,
            "Pred Open": p_open,
            "Pred High": p_high,
            "Pred Low": p_low,
            "Pred Close": p_close,
            "Actual Close": actual_close,
            "Abs Error": abs(p_close - actual_close) if not np.isnan(actual_close) else np.nan,
            "APE %": (abs(p_close - actual_close) / actual_close * 100.0) if actual_close else np.nan,
            "Note": note,
        })

    bt = pd.DataFrame(rows).set_index("Date").sort_index()
    return bt


def run_grid_search(oil_daily: pd.DataFrame, n_days: int,
                    rsi_list: list[int], atr_list: list[int], atrm_list: list[float],
                    ema_list: list[int], volz_list: list[int]) -> pd.DataFrame:
    results = []
    for r in rsi_list:
        for a in atr_list:
            for m in atrm_list:
                for e in ema_list:
                    for v in volz_list:
                        params = ForecastParams(rsi_period=r, atr_period=a, atr_mult=m,
                                                 ema_ret_span=e, vol_z_lookback=v)
                        bt = backtest_last_n_days(oil_daily, n_days=n_days, params=params)
                        if bt.empty or not bt["Abs Error"].notna().any():
                            continue
                        mae = float(bt["Abs Error"].mean())
                        mape = float(bt["APE %"].mean()) if bt["APE %"].notna().any() else np.nan
                        coverage = float(((bt["Actual Close"] >= bt["Pred Low"]) & (bt["Actual Close"] <= bt["Pred High"]))
                                         .mean())
                        results.append({
                            "rsi": r, "atr": a, "atr_mult": m, "ema_span": e, "volz": v,
                            "MAE": mae, "MAPE": mape, "Coverage": coverage, "Samples": len(bt)
                        })
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df["MAPE_fill"] = df["MAPE"].fillna(df["MAPE"].median())
    df = df.sort_values(["MAPE_fill", "MAE", "Coverage"], ascending=[True, True, False])
    df = df.drop(columns=["MAPE_fill"]) 
    return df

# ---------------- Plots ----------------

def fig_forecast_path(daily: pd.DataFrame, preds: List[ForecastPoint]) -> go.Figure:
    hist = daily.tail(60)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Historical Close"))
    if preds:
        fig.add_trace(go.Scatter(x=[p.date for p in preds], y=[p.pred_close for p in preds],
                                 mode="lines+markers", name="Forecast Close"))
        fig.add_trace(go.Scatter(x=[p.date for p in preds], y=[p.pred_high for p in preds],
                                 mode="lines", name="Pred High", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=[p.date for p in preds], y=[p.pred_low for p in preds],
                                 mode="lines", name="Pred Low", line=dict(dash="dot")))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=420)
    return fig


def fig_intraday_15m(df15: pd.DataFrame) -> go.Figure:
    df = df15.copy()
    df["RSI"] = rsi(df["Close"], 14)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.55, 0.15, 0.30])
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 name="Oil 15m"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.5), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI(14)", mode="lines"), row=3, col=1)
    fig.add_hline(y=70, line_width=1, line_dash="dot", row=3, col=1)
    fig.add_hline(y=30, line_width=1, line_dash="dot", row=3, col=1)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=700, xaxis_rangeslider_visible=False)
    return fig


def intraday_summary(df15: pd.DataFrame) -> list[str]:
    out: list[str] = []
    df = df15.copy()
    df["RSI"] = rsi(df["Close"], 14)
    over = int((df["RSI"] > 70).sum())
    under = int((df["RSI"] < 30).sum())
    out.append(f"RSI>70 bars: {over}, RSI<30 bars: {under} (last {len(df)} bars)")
    top_vol = df.nlargest(3, "Volume")
    for i, (ts, row) in enumerate(top_vol.iterrows(), 1):
        out.append(f"Vol spike {i}: {ts.strftime('%Y-%m-%d %H:%M')} — {int(row['Volume']):,}")
    if len(df) > 1:
        drift = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1.0
        out.append(f"Period drift: {drift:+.2%}")
    return out


def build_forecast_table(preds: List[ForecastPoint]) -> pd.DataFrame:
    if not preds:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "Date": p.date.date(),
            "Pred Open": round(p.pred_open, 3),
            "Pred High": round(p.pred_high, 3),
            "Pred Low": round(p.pred_low, 3),
            "Pred Close": round(p.pred_close, 3),
            "Notes": p.note,
        }
        for p in preds
    ])


st.set_page_config(page_title="Oil TA Forecast (RSI + Volume)", layout="wide")
st.title("🛢️ Oil TA Forecast (RSI + Volume)")
st.caption("Data: Yahoo Finance via yfinance. Times: Europe/London. Not investment advice.")

with st.sidebar:
    st.header("Settings")
    rsi_period = st.slider("RSI Period", 5, 30, 14, 1, key="rsi_k")
    atr_period = st.slider("ATR Period", 5, 30, 14, 1, key="atr_k")
    atr_mult = st.slider("ATR Multiplier for High/Low Bands", 0.25, 2.0, 0.75, 0.05, key="atrm_k")
    ema_span = st.slider("EMA Drift Span (days)", 5, 60, 20, 1, key="ema_k")
    vol_look = st.slider("Volume Z-score Lookback", 10, 60, 20, 1, key="volz_k")
    horizon = st.slider("Forecast Horizon (trading days)", 3, 10, 7, 1, key="hzn_k")
    intraday_days = st.slider("Intraday Window (days, 15m)", 2, 30, 10, 1, key="i15_k")
    st.markdown("---")
    backtest_n = st.slider("Backtest Window (days)", 20, 180, 60, 5, key="bt_k")

    with st.expander("Auto‑tune (Grid Search)"):
        st.caption("Search grid; rank by MAPE → MAE → Coverage")
        rsi_grid = st.multiselect("RSI period", [10, 12, 14, 16, 20], default=[10, 14, 20], key="g_rsi")
        atr_grid = st.multiselect("ATR period", [10, 14, 20], default=[10, 14, 20], key="g_atr")
        atrm_grid = st.multiselect("ATR mult", [0.5, 0.75, 1.0], default=[0.5, 0.75, 1.0], key="g_atrm")
        ema_grid = st.multiselect("EMA span", [10, 20, 30, 40], default=[10, 20, 40], key="g_ema")
        volz_grid = st.multiselect("Vol z lookback", [10, 20, 30, 40], default=[10, 20, 40], key="g_volz")
        run_sweep = st.button("Run Sweep", type="primary")
        st.session_state.setdefault("grid_results", None)
        if run_sweep:
            with st.spinner("Running parameter sweep..."):
                st.session_state.grid_results = run_grid_search(
                    oil_daily=load_yf(OIL_TICKER, period="2y", interval="1d"),
                    n_days=backtest_n,
                    rsi_list=rsi_grid,
                    atr_list=atr_grid,
                    atrm_list=atrm_grid,
                    ema_list=ema_grid,
                    volz_list=volz_grid,
                )
        if st.session_state.grid_results is not None and not st.session_state.grid_results.empty:
            top = st.session_state.grid_results.head(10)
            st.dataframe(top, use_container_width=True)
            best = top.iloc[0]
            if st.button("Apply Best & Re‑run"):
                st.session_state.rsi_k = int(best["rsi"]) 
                st.session_state.atr_k = int(best["atr"]) 
                st.session_state.atrm_k = float(best["atr_mult"]) 
                st.session_state.ema_k = int(best["ema_span"]) 
                st.session_state.volz_k = int(best["volz"]) 
                st.rerun()

    st.markdown("---")
    st.write("Ticker:")
    st.code(f"Oil: {OIL_TICKER}")

# Data
with st.spinner("Loading market data from Yahoo..."):
    oil_daily = load_yf(OIL_TICKER, period="2y", interval="1d")
    oil_15m = load_yf(OIL_TICKER, period=f"{intraday_days}d", interval="15m")

if oil_daily.empty:
    st.error("Failed to load daily data. Please retry later.")
    st.stop()

latest_oil = oil_daily.iloc[-1]
cols = st.columns(3)
with cols[0]:
    st.metric("Oil Close (CL=F)", f"{latest_oil['Close']:.2f}",
              delta=f"{(latest_oil['Close']/oil_daily['Close'].iloc[-2]-1):+.2%}")
with cols[1]:
    st.metric("Last Daily Bar", to_london(oil_daily.index[-1]).strftime("%Y-%m-%d"))
with cols[2]:
    st.metric("Records Loaded", f"{len(oil_daily):,}")

# Forecast
params = ForecastParams(
    rsi_period=rsi_period,
    atr_period=atr_period,
    atr_mult=atr_mult,
    ema_ret_span=ema_span,
    vol_z_lookback=vol_look,
)

preds = ta_forecast_next_days(oil_daily, horizon=horizon, params=params)
forecast_df = build_forecast_table(preds)

left, right = st.columns([1.1, 1])
with left:
    st.subheader("7‑Day Prediction Table (High/Low)")
    if forecast_df.empty:
        st.info("Not enough data to forecast.")
    else:
        st.dataframe(forecast_df, use_container_width=True)
        st.download_button(
            "Download CSV",
            forecast_df.to_csv(index=False),
            file_name="oil_forecast_7d.csv",
            mime="text/csv",
        )

with right:
    st.subheader("Forecast Path vs Recent History")
    st.plotly_chart(fig_forecast_path(oil_daily, preds), use_container_width=True, theme="streamlit")

# Backtest panel
st.markdown("---")
st.subheader("Backtest — Last N Days (1‑Step Ahead)")
with st.spinner("Running backtest..."):
    bt = backtest_last_n_days(oil_daily, n_days=backtest_n, params=params)

if bt.empty:
    st.info("Not enough history to run backtest.")
else:
    err = bt["Pred Close"] - bt["Actual Close"]
    mae = float(bt["Abs Error"].mean())
    mape = float(bt["APE %"].mean()) if bt["APE %"].notna().any() else float("nan")
    coverage = float(((bt["Actual Close"] >= bt["Pred Low"]) & (bt["Actual Close"] <= bt["Pred High"]))
                     .mean())
    bias = float(err.mean())
    stdev = float(err.std(ddof=1))

    cols_bt = st.columns(6)
    with cols_bt[0]:
        st.metric("MAE", f"{mae:.3f}")
    with cols_bt[1]:
        st.metric("MAPE", f"{mape:.2f}%")
    with cols_bt[2]:
        st.metric("Band Coverage", f"{coverage:.0%}")
    with cols_bt[3]:
        st.metric("Bias (Mean Error)", f"{bias:+.3f}")
    with cols_bt[4]:
        st.metric("StDev(Error)", f"{stdev:.3f}")
    with cols_bt[5]:
        st.metric("Samples", f"{len(bt)}")

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt.index, y=bt["Actual Close"], mode="lines", name="Actual Close"))
    fig_bt.add_trace(go.Scatter(x=bt.index, y=bt["Pred Close"], mode="lines+markers", name="Pred Close"))
    fig_bt.add_trace(go.Scatter(x=bt.index, y=bt["Pred High"], mode="lines", name="Pred High", line=dict(dash="dot")))
    fig_bt.add_trace(go.Scatter(x=bt.index, y=bt["Pred Low"], mode="lines", name="Pred Low", line=dict(dash="dot")))
    fig_bt.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=420)
    st.plotly_chart(fig_bt, use_container_width=True, theme="streamlit")

    st.markdown("**Residual Diagnostics**")
    diag_cols = st.columns(3)

    with diag_cols[0]:
        hist = go.Figure()
        hist.add_trace(go.Histogram(x=err.dropna(), nbinsx=30, name="Errors"))
        hist.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=280)
        st.plotly_chart(hist, use_container_width=True, theme="streamlit")

    with diag_cols[1]:
        if _stats is not None:
            e = np.sort(err.dropna().values)
            n = len(e)
            if n > 2:
                probs = (np.arange(1, n + 1) - 0.5) / n
                qn = _stats.norm.ppf(probs)
                qq = go.Figure()
                qq.add_trace(go.Scatter(x=qn, y=e, mode="markers", name="QQ"))
                m, c = np.polyfit(qn, e, 1)
                line_x = np.array([qn.min(), qn.max()])
                qq.add_trace(go.Scatter(x=line_x, y=m*line_x + c, mode="lines", name="Fit", line=dict(dash="dot")))
                qq.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=280)
                st.plotly_chart(qq, use_container_width=True, theme="streamlit")
        else:
            st.info("Install scipy for QQ plot: pip install scipy")

    with diag_cols[2]:
        e = err.dropna().values
        lags = min(10, len(e) - 2) if len(e) > 2 else 0
        acf_vals = []
        if lags > 0:
            e0 = (e - e.mean())
            denom = float(e0 @ e0)
            for k in range(1, lags + 1):
                num = float(e0[:-k] @ e0[k:])
                acf_vals.append(num / denom if denom != 0 else 0.0)
        acf_fig = go.Figure()
        acf_fig.add_trace(go.Bar(x=list(range(1, lags + 1)), y=acf_vals, name="ACF"))
        acf_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=280)
        st.plotly_chart(acf_fig, use_container_width=True, theme="streamlit")

    st.markdown("**Backtest Details**")
    bt_show = bt.copy()
    bt_show.index = bt_show.index.date
    st.dataframe(bt_show, use_container_width=True)
    st.download_button(
        "Download Backtest CSV",
        bt.to_csv(),
        file_name="oil_ta_backtest.csv",
        mime="text/csv",
    )

# Intraday
st.markdown("---")
st.subheader("Intraday 15‑Minute View (Oil)")
if not oil_15m.empty:
    st.plotly_chart(fig_intraday_15m(oil_15m), use_container_width=True, theme="streamlit")
    st.markdown("**Intraday Notes**")
    for b in intraday_summary(oil_15m):
        st.write("• ", b)
else:
    st.info("15m intraday chart unavailable. Adjust 'Intraday Window' or try later.")

# Disclaimer
st.markdown(
    "> **Disclaimer**: Educational tool using heuristic technical analysis (RSI, Volume, ATR). "
    "Not financial advice. Futures trading involves substantial risk."
)
