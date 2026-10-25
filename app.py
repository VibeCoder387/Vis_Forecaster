from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from forecast_engine import (
    MODEL_DIR,
    FEATURES,
    IST,
    UTC,
    load_models,
    fetch_latest_ogimet_metars,
    parse_datetime_value_table,
    run_forecast,
)

# ------------------------------------------------------------
# App configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="VIDP • Visibility Intelligence",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "vidp_forecast_history.sqlite3"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #07111f; color: #e8eef7; }
[data-testid="stHeader"] { background: rgba(7,17,31,.85); }
[data-testid="stSidebar"] { background: #0a1728; border-right: 1px solid #19314e; }
[data-testid="stSidebar"] * { color: #dce7f5; }
.block-container { max-width: 1500px; padding: 2rem 2.4rem 4rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif; color:#f4f8ff; letter-spacing:-.03em; }
.hero { background: radial-gradient(circle at 85% 10%, rgba(31,130,255,.22), transparent 34%), linear-gradient(135deg,#0d2037,#091522 65%); border:1px solid #1d3c5e; border-radius:24px; padding:30px 34px; margin-bottom:20px; box-shadow:0 20px 60px rgba(0,0,0,.25); }
.hero-title { font:700 2.2rem 'Space Grotesk'; margin:0; color:#fff; }
.hero-sub { color:#9eb1c9; margin-top:8px; font-size:1rem; }
.pill { display:inline-block; padding:6px 10px; border-radius:999px; background:#102a45; border:1px solid #214c72; color:#83c8ff; font-size:.78rem; margin-right:6px; }
.card { background:#0c1b2d; border:1px solid #1b3552; border-radius:18px; padding:18px 20px; box-shadow:0 12px 35px rgba(0,0,0,.16); }
.metric-label { color:#8ea4bd; font-size:.78rem; text-transform:uppercase; letter-spacing:.09em; }
.metric-value { font:700 1.65rem 'Space Grotesk'; color:#f7fbff; margin-top:5px; }
.metric-note { color:#6f89a5; font-size:.78rem; margin-top:3px; }
.section { font:600 1.18rem 'Space Grotesk'; color:#eef5ff; margin:24px 0 10px; }
.source { background:#0a1828; border:1px solid #193550; border-radius:15px; padding:14px 16px; }
.source-title { font-weight:700; color:#dcecff; }
.source-ok { color:#56d6a1; } .source-warn { color:#ffca69; } .source-off { color:#8fa2b7; }
[data-testid="stTextArea"] textarea { background:#081524; color:#e8f1fb; border:1px solid #21415f; border-radius:12px; font-family:Consolas,monospace; font-size:.83rem; }
[data-testid="stTextInput"] input { background:#081524; color:#e8f1fb; border:1px solid #21415f; }
.stButton > button[kind="primary"] { background:linear-gradient(135deg,#1685ff,#00b7a8); border:0; border-radius:13px; font-weight:700; padding:.8rem 1rem; }
.stButton > button { border-radius:11px; }
[data-testid="stDataFrame"] { border:1px solid #1b3552; border-radius:14px; overflow:hidden; }
div[data-baseweb="tab-list"] { gap:8px; }
button[data-baseweb="tab"] { background:#0c1b2d; border-radius:10px 10px 0 0; color:#9fb2c8; }
small, .stCaption { color:#7990a9 !important; }
hr { border-color:#18314b; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Persistence and timestamp helpers
# ------------------------------------------------------------
def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                origin_utc TEXT NOT NULL,
                target_utc TEXT NOT NULL,
                step INTEGER NOT NULL,
                predicted_m REAL NOT NULL,
                actual_m REAL,
                created_utc TEXT NOT NULL,
                UNIQUE(origin_utc, step)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                timestamp_utc TEXT PRIMARY KEY,
                visibility_m REAL NOT NULL,
                fetched_utc TEXT NOT NULL
            )
        """)


def as_utc_iso(value) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def display_ist(value) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(IST).strftime("%d %b %Y %H:%M")


def save_observations(metars):
    fetched_at = datetime.now(timezone.utc).isoformat()
    with db_connect() as conn:
        for m in metars:
            conn.execute(
                """
                INSERT INTO observations(timestamp_utc, visibility_m, fetched_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(timestamp_utc) DO UPDATE SET
                    visibility_m=excluded.visibility_m,
                    fetched_utc=excluded.fetched_utc
                """,
                (as_utc_iso(m.timestamp_utc), float(m.visibility_m), fetched_at),
            )


def save_forecasts(forecast, origin_utc):
    created = datetime.now(timezone.utc).isoformat()
    origin = as_utc_iso(origin_utc)
    with db_connect() as conn:
        for _, row in forecast.iterrows():
            conn.execute(
                """
                INSERT INTO forecasts
                    (origin_utc, target_utc, step, predicted_m, created_utc)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(origin_utc, step) DO NOTHING
                """,
                (
                    origin,
                    as_utc_iso(row["forecast_time_utc"]),
                    int(row["step"]),
                    float(row["visibility_m"]),
                    created,
                ),
            )


def reconcile_forecasts():
    """Match saved predictions to the nearest observed METAR within 15 minutes."""
    with db_connect() as conn:
        forecasts = pd.read_sql_query(
            "SELECT id, target_utc FROM forecasts WHERE actual_m IS NULL",
            conn,
        )
        observations = pd.read_sql_query(
            "SELECT timestamp_utc, visibility_m FROM observations",
            conn,
        )

        if forecasts.empty or observations.empty:
            return 0

        observations["timestamp"] = pd.to_datetime(
            observations["timestamp_utc"], utc=True, errors="coerce"
        )
        observations = observations.dropna(subset=["timestamp"])
        obs_times = observations["timestamp"].astype("int64").to_numpy()
        obs_vis = observations["visibility_m"].to_numpy()

        updates = []
        tolerance_ns = 15 * 60 * 1_000_000_000

        for _, forecast_row in forecasts.iterrows():
            target = pd.to_datetime(forecast_row["target_utc"], utc=True, errors="coerce")
            if pd.isna(target):
                continue
            target_ns = target.value
            if len(obs_times) == 0:
                continue
            nearest_idx = int(np.argmin(np.abs(obs_times - target_ns)))
            if abs(int(obs_times[nearest_idx]) - target_ns) <= tolerance_ns:
                updates.append((float(obs_vis[nearest_idx]), int(forecast_row["id"])))

        if updates:
            conn.executemany(
                "UPDATE forecasts SET actual_m=? WHERE id=? AND actual_m IS NULL",
                updates,
            )
        return len(updates)


def load_forecast_history():
    with db_connect() as conn:
        return pd.read_sql_query(
            """
            SELECT origin_utc, target_utc, step, predicted_m, actual_m, created_utc
            FROM forecasts
            ORDER BY target_utc DESC
            """,
            conn,
        )


def load_observation_history():
    with db_connect() as conn:
        return pd.read_sql_query(
            """
            SELECT timestamp_utc, visibility_m, fetched_utc
            FROM observations
            ORDER BY timestamp_utc DESC
            """,
            conn,
        )


@st.cache_resource
def get_models():
    return load_models()


@st.cache_data(ttl=300, show_spinner=False)
def get_latest_metars():
    # Fetch enough recent history to verify recently matured forecasts.
    return fetch_latest_ogimet_metars(hours=72)


init_db()

st.markdown("""
<div class="hero">
  <div><span class="pill">VIDP</span><span class="pill">MARK-2</span><span class="pill">24 × LIGHTGBM</span><span class="pill">12 HOURS</span></div>
  <div class="hero-title">Visibility Intelligence Console</div>
  <div class="hero-sub">Live METAR ingestion, 12-hour visibility forecasting and continuous forecast verification.</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ✈ Model Console")
    try:
        models = get_models()
        st.success("24 / 24 models ready")
        st.caption(f"Native LightGBM • {len(FEATURES)} features")
        st.caption(f"Folder: `{MODEL_DIR.name}`")
    except Exception as e:
        models = None
        st.error("Model load failed")
        st.exception(e)

    st.markdown("---")
    st.markdown("**Time policy**")
    st.caption("METAR observations: UTC")
    st.caption("PM2.5 / PBLH inputs: IST")
    st.caption("Dashboard + forecast: IST")
    st.caption("Model time features: UTC, matching training")

    st.markdown("---")
    st.markdown("**Verification policy**")
    st.caption("A prediction is scored only after a matching observation is available.")
    st.caption("Observed METAR must be within 15 minutes of the forecast target.")
    st.caption("SQLite history is local to this deployment.")

# ------------------------------------------------------------
# Fetch current observations automatically
# ------------------------------------------------------------
st.markdown(
    '<div class="section">1 · Live METAR observations</div>',
    unsafe_allow_html=True,
)

refresh_col, status_col = st.columns([1, 4])
with refresh_col:
    refresh = st.button("↻ Refresh METARs")
if refresh:
    get_latest_metars.clear()

metars = None
metar_error = None
try:
    with st.spinner("Fetching recent VIDP METAR observations from Ogimet..."):
        metars = get_latest_metars()
    save_observations(metars)
    newly_verified = reconcile_forecasts()
    st.success(
        f"Retrieved {len(metars)} usable METAR observations. "
        f"Forecasts newly verified this refresh: {newly_verified}."
    )
except Exception as e:
    metar_error = str(e)
    st.error(f"Could not retrieve live Ogimet observations: {metar_error}")

if metars:
    current = metars[-1]
    metric_cols = st.columns(5)
    latest_metrics = [
        ("Latest visibility", f"{current.visibility_m:,.0f} m"),
        ("Observation time", current.timestamp_ist.strftime("%d %b • %H:%M IST")),
        ("Wind", f"{current.wind:g} kt" if current.wind is not None and pd.notna(current.wind) else "N/A"),
        ("Temperature", f"{current.temp:g} °C" if current.temp is not None and pd.notna(current.temp) else "N/A"),
        ("Dew point", f"{current.dew:g} °C" if current.dew is not None and pd.notna(current.dew) else "N/A"),
    ]
    for col, (label, value) in zip(metric_cols, latest_metrics):
        col.markdown(
            f'<div class="card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    parsed_rows = []
    for m in reversed(metars[-48:]):
        parsed_rows.append({
            "Observation time (IST)": m.timestamp_ist.strftime("%Y-%m-%d %H:%M"),
            "Observation time (UTC)": m.timestamp_utc.strftime("%Y-%m-%d %H:%M"),
            "Visibility (m)": m.visibility_m,
            "Wind (kt)": m.wind,
            "Temperature (°C)": m.temp,
            "Dew point (°C)": m.dew,
            "Fog (FG)": m.fog,
            "Mist (BR)": m.mist,
        })

    with st.expander("View parsed METAR data", expanded=True):
        st.caption("Most recent 48 observations, newest first. Times are shown in both UTC and IST.")
        st.dataframe(
            pd.DataFrame(parsed_rows),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.warning("Live parsed observations are unavailable. Forecast inference is disabled until Ogimet data can be fetched.")

# ------------------------------------------------------------
# Optional atmospheric context
# ------------------------------------------------------------
st.markdown(
    '<div class="section">2 · Atmospheric context <span style="color:#6f89a5;font-size:.8rem">optional sources • independent timelines</span></div>',
    unsafe_allow_html=True,
)
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        '<div class="source"><div class="source-title">PM2.5 · IST</div>'
        '<div class="source-ok">Optional • latest available value is used</div></div>',
        unsafe_allow_html=True,
    )
    pm25_text = st.text_area(
        "PM2.5 observations",
        "2026-09-26 06:00, 72\n2026-09-26 07:00, 75\n2026-09-26 08:00, 78\n2026-09-26 09:00, 81\n2026-09-26 10:00, 80\n2026-09-26 11:00, 82\n2026-09-26 12:00, 84\n2026-09-26 13:00, 85",
        height=190,
        label_visibility="collapsed",
    )

with col2:
    st.markdown(
        '<div class="source"><div class="source-title">PBLH / Inversion · IST</div>'
        '<div class="source-ok">Optional • each field aligned independently</div></div>',
        unsafe_allow_html=True,
    )
    inv_height_text = st.text_area("inv_height", "2026-09-26 06:00, 180\n2026-09-26 12:00, 160", height=82, label_visibility="collapsed")
    inv_strength_text = st.text_area("inv_strength_k", "2026-09-26 06:00, 0.8\n2026-09-26 12:00, 0.9", height=82, label_visibility="collapsed")
    pblh_temp_text = st.text_area("pblh_temp_c", "2026-09-26 06:00, 24.0\n2026-09-26 12:00, 23.5", height=82, label_visibility="collapsed")
    inv_top_temp_text = st.text_area("inv_top_temp_c", "2026-09-26 06:00, 20.0\n2026-09-26 12:00, 19.5", height=82, label_visibility="collapsed")
    inv_2_layer_text = st.text_area("inv_2_layer", "2026-09-26 06:00, 0\n2026-09-26 12:00, 1", height=82, label_visibility="collapsed")

# ------------------------------------------------------------
# Forecast inference
# ------------------------------------------------------------
st.markdown(
    '<div class="section">3 · Run inference</div>',
    unsafe_allow_html=True,
)
run = st.button("🚀 Generate 12-hour forecast", type="primary", use_container_width=True)

if run:
    try:
        if models is None:
            raise RuntimeError("The 24 LightGBM models are not loaded.")
        if not metars:
            raise RuntimeError("No usable live Ogimet METAR data is available.")

        pm25 = parse_datetime_value_table(pm25_text, "pm25")
        inv_height = parse_datetime_value_table(inv_height_text, "inv_height")
        inv_strength = parse_datetime_value_table(inv_strength_text, "inv_strength_k")
        pblh_temp = parse_datetime_value_table(pblh_temp_text, "pblh_temp_c")
        inv_top_temp = parse_datetime_value_table(inv_top_temp_text, "inv_top_temp_c")
        inv_2_layer = parse_datetime_value_table(inv_2_layer_text, "inv_2_layer")

        forecast, feature_row, status = run_forecast(
            metars,
            pm25,
            inv_height,
            inv_strength,
            pblh_temp,
            inv_top_temp,
            inv_2_layer,
            models,
        )

        current = metars[-1]
        origin_ist = current.timestamp_ist

        # Persist the forecast so it can be verified by later observations.
        save_forecasts(forecast, current.timestamp_utc)
        reconcile_forecasts()

        st.markdown('<div class="section">Forecast overview</div>', unsafe_allow_html=True)
        cards = st.columns(5)
        metrics = [
            ("Current visibility", f"{current.visibility_m:,.0f} m", "latest METAR"),
            ("Forecast origin", origin_ist.strftime("%d %b • %H:%M IST"), "METAR UTC converted"),
            ("+30 min", f"{forecast.iloc[0].visibility_m:,.0f} m", "calibrated + rounded"),
            ("12-hour minimum", f"{forecast.visibility_m.min():,.0f} m", "across 24 rounded steps"),
            ("PM2.5", f"{status['pm25_missing_features']}/6 missing", "optional feature set"),
        ]
        for c, (label, value, note) in zip(cards, metrics):
            c.markdown(
                f'<div class="card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True,
            )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=forecast["forecast_time"],
                y=forecast["visibility_m"],
                mode="lines+markers",
                name="Visibility",
                line=dict(width=3),
            )
        )
        fig.add_hline(y=5000, line_dash="dash", annotation_text="5 km", annotation_position="top left")
        fig.add_hline(y=1000, line_dash="dot", annotation_text="1 km", annotation_position="top left")
        fig.update_layout(
            height=470,
            template="plotly_dark",
            paper_bgcolor="#0c1b2d",
            plot_bgcolor="#0c1b2d",
            margin=dict(l=20, r=20, t=35, b=20),
            hovermode="x unified",
            xaxis_title="Forecast time (IST)",
            yaxis_title="Visibility (m)",
            font=dict(color="#cbd8e8"),
            legend=dict(orientation="h", y=1.08, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["Forecast table", "Source health", "39-feature audit"])
        with tab1:
            display = forecast.copy()
            display["forecast_time"] = pd.to_datetime(display["forecast_time"]).dt.strftime("%d %b %Y %H:%M")
            display["forecast_time_utc"] = pd.to_datetime(display["forecast_time_utc"], utc=True).dt.strftime("%d %b %Y %H:%M")
            for col in ["visibility_model_m", "visibility_calibrated_m", "visibility_m"]:
                display[col] = display[col].round(0).astype(int)
            display["visibility_km"] = display["visibility_km"].round(3)
            st.dataframe(
                display[["step", "minutes_ahead", "forecast_time", "visibility_model_m", "visibility_calibrated_m", "visibility_m", "calibration_method"]],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download IST forecast CSV",
                display.to_csv(index=False).encode(),
                f"VIDP_forecast_{origin_ist:%Y%m%d_%H%M}_IST.csv",
                "text/csv",
            )

        with tab2:
            s1, s2, s3 = st.columns(3)
            s1.metric("METAR", "READY", "Ogimet UTC → IST display")
            s2.metric("PM2.5", "AVAILABLE" if status["pm25_current_available"] else "MISSING", "missing is allowed")
            s3.metric("PBLH / inversion", "AVAILABLE" if status["pblh_available"] else "PARTIAL", "missing fields stay NaN")
            st.info(
                "Optional sources are never allowed to stop METAR-based inference. "
                "Missing numeric inputs remain NaN and are handled by LightGBM."
            )

        with tab3:
            audit = feature_row.T.reset_index()
            audit.columns = ["feature", "value"]
            audit["value"] = audit["value"].apply(
                lambda x: "NaN (source unavailable)" if pd.isna(x) else x
            )
            st.dataframe(audit, use_container_width=True, hide_index=True)

        methods = sorted(set(forecast["calibration_method"]))
        st.success(
            f"Forecast generated and saved • origin {origin_ist:%d %b %Y %H:%M IST} "
            f"• +30 min to +12 h • calibration: {', '.join(methods)}"
        )

    except Exception as e:
        st.error(f"Forecast generation failed: {e}")
        with st.expander("Technical details"):
            st.exception(e)

# ------------------------------------------------------------
# Forecast verification dashboard
# ------------------------------------------------------------
st.markdown(
    '<div class="section">4 · Forecast accuracy & verification</div>',
    unsafe_allow_html=True,
)
history = load_forecast_history()

if history.empty:
    st.info(
        "No saved forecasts yet. Generate a forecast and return after its target times "
        "have passed and matching METAR observations have arrived. Accuracy will build over time."
    )
else:
    verified = history.dropna(subset=["actual_m"]).copy()
    total = len(history)
    verified_count = len(verified)
    pending_count = total - verified_count

    if verified_count:
        verified["error_m"] = verified["predicted_m"] - verified["actual_m"]
        verified["absolute_error_m"] = verified["error_m"].abs()
        mae = float(verified["absolute_error_m"].mean())
        rmse = float(np.sqrt(np.mean(np.square(verified["error_m"]))))
        bias = float(verified["error_m"].mean())
    else:
        mae = rmse = bias = np.nan

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Verified predictions", f"{verified_count:,}", f"{pending_count:,} awaiting observations")
    k2.metric("MAE", f"{mae:,.0f} m" if np.isfinite(mae) else "—")
    k3.metric("RMSE", f"{rmse:,.0f} m" if np.isfinite(rmse) else "—")
    k4.metric("Bias", f"{bias:+,.0f} m" if np.isfinite(bias) else "—", "positive = overprediction" if np.isfinite(bias) else None)

    if verified_count:
        by_step = (
            verified.assign(
                error=verified["predicted_m"] - verified["actual_m"],
                abs_error=lambda d: d["error"].abs(),
            )
            .groupby("step", as_index=False)
            .agg(
                samples=("error", "count"),
                MAE_m=("abs_error", "mean"),
                Bias_m=("error", "mean"),
            )
        )
        by_step["lead_time_min"] = by_step["step"] * 30
        fig_acc = go.Figure()
        fig_acc.add_trace(
            go.Scatter(
                x=by_step["lead_time_min"],
                y=by_step["MAE_m"],
                mode="lines+markers",
                name="MAE",
            )
        )
        fig_acc.update_layout(
            height=330,
            template="plotly_dark",
            paper_bgcolor="#0c1b2d",
            plot_bgcolor="#0c1b2d",
            margin=dict(l=20, r=20, t=25, b=20),
            xaxis_title="Forecast lead time (minutes)",
            yaxis_title="MAE (m)",
            font=dict(color="#cbd8e8"),
        )
        st.plotly_chart(fig_acc, use_container_width=True)
        with st.expander("Accuracy by forecast lead time"):
            st.dataframe(
                by_step[["lead_time_min", "samples", "MAE_m", "Bias_m"]].round(2),
                use_container_width=True,
                hide_index=True,
            )

        recent = verified.copy()
        recent["Forecast issued (IST)"] = recent["origin_utc"].apply(display_ist)
        recent["Target time (IST)"] = recent["target_utc"].apply(display_ist)
        recent["Absolute error (m)"] = (recent["predicted_m"] - recent["actual_m"]).abs()
        recent = recent.rename(
            columns={"step": "Step", "predicted_m": "Predicted (m)", "actual_m": "Observed (m)"}
        )
        st.dataframe(
            recent[["Forecast issued (IST)", "Target time (IST)", "Step", "Predicted (m)", "Observed (m)", "Absolute error (m)"]]
            .sort_values("Target time (IST)", ascending=False)
            .head(100),
            use_container_width=True,
            hide_index=True,
        )

    pending = history[history["actual_m"].isna()].copy()
    if not pending.empty:
        with st.expander(f"Predictions awaiting verification ({len(pending)})"):
            pending["Forecast issued (IST)"] = pending["origin_utc"].apply(display_ist)
            pending["Target time (IST)"] = pending["target_utc"].apply(display_ist)
            pending = pending.rename(columns={"step": "Step", "predicted_m": "Predicted (m)"})
            st.dataframe(
                pending[["Forecast issued (IST)", "Target time (IST)", "Step", "Predicted (m)"]].head(100),
                use_container_width=True,
                hide_index=True,
            )

    csv_history = history.copy()
    csv_history["origin_ist"] = csv_history["origin_utc"].apply(display_ist)
    csv_history["target_ist"] = csv_history["target_utc"].apply(display_ist)
    st.download_button(
        "Download forecast verification history",
        csv_history.to_csv(index=False).encode("utf-8"),
        "VIDP_forecast_verification_history.csv",
        "text/csv",
    )

with st.expander("Stored METAR observations"):
    observations = load_observation_history()
    if observations.empty:
        st.caption("No observations stored yet.")
    else:
        observations["Observation time (IST)"] = observations["timestamp_utc"].apply(display_ist)
        observations = observations.rename(columns={"visibility_m": "Observed visibility (m)"})
        st.dataframe(
            observations[["Observation time (IST)", "Observed visibility (m)", "fetched_utc"]].head(200),
            use_container_width=True,
            hide_index=True,
        )

st.markdown("---")
st.caption(
    "VIDP Mark-2 inference console • 24 native LightGBM models • exact 39-feature schema • "
    "no retraining performed by the app. Accuracy metrics use forecasts saved by this app, "
    "matched to later METAR observations."
)
