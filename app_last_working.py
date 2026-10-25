from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from forecast_engine import (
    MODEL_DIR, FEATURES, IST, UTC, load_models, parse_metar_block,fetch_latest_ogimet_metars,
    parse_datetime_value_table, run_forecast,
)

st.set_page_config(
    page_title="VIDP • Visibility Intelligence",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

@st.cache_resource
def get_models():
    return load_models()

st.markdown("""
<div class="hero">
  <div><span class="pill">VIDP</span><span class="pill">MARK-2</span><span class="pill">24 × LIGHTGBM</span><span class="pill">12 HOURS</span></div>
  <div class="hero-title">Visibility Intelligence Console</div>
  <div class="hero-sub">Aviation visibility forecasting from METAR history, PM2.5 and boundary-layer signals — with UTC-safe ingestion and IST-first presentation.</div>
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
    st.caption("METAR input: UTC")
    st.caption("PM2.5 / PBLH input: IST")
    st.caption("Dashboard + forecast: IST")
    st.caption("Model time features: UTC, matching training")

    st.markdown("---")
    st.markdown("**Missing-source policy**")
    st.caption("PM2.5 and PBLH are optional at inference. Missing numeric features are passed as NaN; LightGBM handles missing values natively.")

st.markdown('<div class="section">1 · METAR history <span style="color:#6f89a5;font-size:.8rem">UTC input • visibility extracted automatically</span></div>', unsafe_allow_html=True)
st.info("Paste 13 or more METAR observations. The first 12 previous visibility observations build the lag features; the latest METAR is the forecast origin. Do not enter visibility separately.")

default_metars = """202609260700 METAR VIDP 260700Z 03008KT 5000 BR SCT025 BKN080 24/21 Q1005 NOSIG=
202609260730 METAR VIDP 260730Z 03009KT 4500 BR SCT025 BKN080 24/21 Q1005 NOSIG=
202609260800 METAR VIDP 260800Z 03009KT 4000 BR SCT025 BKN080 24/21 Q1005 NOSIG=
202609260830 METAR VIDP 260830Z 35010KT 3500 BR SCT025 BKN080 24/21 Q1005 NOSIG=
202609260900 METAR VIDP 260900Z 35010KT 3000 BR SCT025 BKN080 23/21 Q1004 NOSIG=
202609260930 METAR VIDP 260930Z 03010KT 2800 BR SCT025 BKN080 23/21 Q1004 NOSIG=
202609261000 METAR VIDP 261000Z 03010KT 3000 BR SCT025 BKN080 23/21 Q1004 NOSIG=
202609261030 METAR VIDP 261030Z 03010KT 3500 BR SCT025 BKN080 23/21 Q1004 NOSIG=
202609261100 METAR VIDP 261100Z 03010KT 4000 BR SCT025 BKN080 23/21 Q1004 NOSIG=
202609261130 METAR VIDP 261130Z 03010KT 4500 BR SCT025 BKN080 23/21 Q1004 NOSIG=
202609261200 METAR VIDP 261200Z 03010KT 4800 BR SCT025 BKN080 23/21 Q1004 NOSIG=
202609261230 METAR VIDP 261230Z 03010KT 5000 BR SCT025 BKN080 23/21 Q1004 NOSIG=
202609261300 METAR VIDP 261300Z 03010KT 5000 BR SCT025 BKN080 23/21 Q1004 NOSIG="""
metar_text = st.text_area("METAR observations", default_metars, height=260)

st.markdown('<div class="section">2 · Atmospheric context <span style="color:#6f89a5;font-size:.8rem">optional sources • independent timelines</span></div>', unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown('<div class="source"><div class="source-title">PM2.5 · IST</div><div class="source-ok">Optional • latest available value is used</div></div>', unsafe_allow_html=True)
    pm25_text = st.text_area("PM2.5 observations", "2026-09-26 06:00, 72\n2026-09-26 07:00, 75\n2026-09-26 08:00, 78\n2026-09-26 09:00, 81\n2026-09-26 10:00, 80\n2026-09-26 11:00, 82\n2026-09-26 12:00, 84\n2026-09-26 13:00, 85", height=190, label_visibility="collapsed")
with col2:
    st.markdown('<div class="source"><div class="source-title">PBLH / Inversion · IST</div><div class="source-ok">Optional • each field aligned independently</div></div>', unsafe_allow_html=True)
    inv_height_text = st.text_area("inv_height", "2026-09-26 06:00, 180\n2026-09-26 12:00, 160", height=82, label_visibility="collapsed")
    inv_strength_text = st.text_area("inv_strength_k", "2026-09-26 06:00, 0.8\n2026-09-26 12:00, 0.9", height=82, label_visibility="collapsed")
    pblh_temp_text = st.text_area("pblh_temp_c", "2026-09-26 06:00, 24.0\n2026-09-26 12:00, 23.5", height=82, label_visibility="collapsed")
    inv_top_temp_text = st.text_area("inv_top_temp_c", "2026-09-26 06:00, 20.0\n2026-09-26 12:00, 19.5", height=82, label_visibility="collapsed")
    inv_2_layer_text = st.text_area("inv_2_layer", "2026-09-26 06:00, 0\n2026-09-26 12:00, 1", height=82, label_visibility="collapsed")

st.markdown('<div class="section">3 · Run inference</div>', unsafe_allow_html=True)
run = st.button("🚀  Generate 12-hour forecast", type="primary", use_container_width=True)

if run:
    try:
        if models is None:
            raise RuntimeError("The 24 LightGBM models are not loaded.")

        # metars = parse_metar_block(metar_text)
        metars = fetch_latest_ogimet_metars(hours=24)
        pm25 = parse_datetime_value_table(pm25_text, "pm25")
        inv_height = parse_datetime_value_table(inv_height_text, "inv_height")
        inv_strength = parse_datetime_value_table(inv_strength_text, "inv_strength_k")
        pblh_temp = parse_datetime_value_table(pblh_temp_text, "pblh_temp_c")
        inv_top_temp = parse_datetime_value_table(inv_top_temp_text, "inv_top_temp_c")
        inv_2_layer = parse_datetime_value_table(inv_2_layer_text, "inv_2_layer")

        forecast, feature_row, status = run_forecast(
            metars, pm25, inv_height, inv_strength, pblh_temp, inv_top_temp, inv_2_layer, models
        )
        current = metars[-1]
        origin_ist = current.timestamp_ist

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
            c.markdown(f'<div class="card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast["forecast_time"], y=forecast["visibility_m"], mode="lines+markers", name="Visibility", line=dict(width=3)))
        fig.add_hline(y=5000, line_dash="dash", annotation_text="5 km", annotation_position="top left")
        fig.add_hline(y=1000, line_dash="dot", annotation_text="1 km", annotation_position="top left")
        fig.update_layout(
            height=470, template="plotly_dark", paper_bgcolor="#0c1b2d", plot_bgcolor="#0c1b2d",
            margin=dict(l=20,r=20,t=35,b=20), hovermode="x unified",
            xaxis_title="Forecast time (IST)", yaxis_title="Visibility (m)",
            font=dict(color="#cbd8e8"), legend=dict(orientation="h", y=1.08, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["Forecast table", "Source health", "39-feature audit"])
        with tab1:
            display = forecast.copy()
            display["forecast_time"] = display["forecast_time"].dt.strftime("%d %b %Y %H:%M")
            display["forecast_time_utc"] = display["forecast_time_utc"].dt.strftime("%d %b %Y %H:%M")
            display["visibility_model_m"] = display["visibility_model_m"].round(0).astype(int)
            display["visibility_calibrated_m"] = display["visibility_calibrated_m"].round(0).astype(int)
            display["visibility_m"] = display["visibility_m"].astype(int)
            display["visibility_km"] = display["visibility_km"].round(3)
            st.dataframe(display[["step","minutes_ahead","forecast_time","visibility_model_m","visibility_calibrated_m","visibility_m","calibration_method"]], use_container_width=True, hide_index=True)
            st.download_button("Download IST forecast CSV", display.to_csv(index=False).encode(), f"VIDP_forecast_{origin_ist:%Y%m%d_%H%M}_IST.csv", "text/csv")
        with tab2:
            s1,s2,s3 = st.columns(3)
            s1.metric("METAR", "READY", "UTC → IST display")
            s2.metric("PM2.5", "AVAILABLE" if status["pm25_current_available"] else "MISSING", "missing is allowed")
            s3.metric("PBLH / inversion", "AVAILABLE" if status["pblh_available"] else "PARTIAL", "missing fields stay NaN")
            st.info("Optional sources are never allowed to stop METAR-based inference. Missing numeric inputs remain NaN and are handled by native LightGBM missing-value routing.")
        with tab3:
            audit = feature_row.T.reset_index(); audit.columns=["feature","value"]
            audit["value"] = audit["value"].apply(lambda x: "NaN (source unavailable)" if pd.isna(x) else x)
            st.dataframe(audit, use_container_width=True, hide_index=True)

        with st.expander("METAR parsing & timezone audit"):
            audit_rows=[]
            for m in metars[-5:]:
                audit_rows.append({"METAR UTC":m.timestamp_utc.strftime("%Y-%m-%d %H:%M"), "Dashboard IST":m.timestamp_ist.strftime("%Y-%m-%d %H:%M"), "Visibility (m)":m.visibility_m, "Wind (kt)":m.wind, "Temp (°C)":m.temp, "Dew (°C)":m.dew, "FG":m.fog, "BR":m.mist})
            st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

        methods = sorted(set(forecast["calibration_method"]))
        st.success(f"Forecast generated • origin {origin_ist:%d %b %Y %H:%M IST} • +30 min to +12 h • calibration: {', '.join(methods)}")

    except Exception as e:
        st.error(f"Forecast generation failed: {e}")
        with st.expander("Technical details"):
            st.exception(e)

st.markdown("---")
st.caption("VIDP Mark-2 inference console • 24 native LightGBM models • exact 39-feature schema • no retraining performed by the app")
