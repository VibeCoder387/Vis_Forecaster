VIDP Visibility Intelligence Console — Mark 2

WHAT THIS VERSION FIXES
1. METAR timestamps are explicitly treated as UTC.
2. PM2.5 and PBLH/inversion timestamps are explicitly treated as IST.
3. METAR UTC timestamps are converted to IST for dashboard display and for cross-source as-of alignment.
4. The trained model's seven time features remain based on UTC because that matches the original training code.
5. PM2.5 and PBLH are independent optional sources. Missing values do not stop inference.
6. Missing optional numeric features are passed as NaN so native LightGBM missing-value handling can route them.
7. Forecast times are displayed in IST; the CSV also contains UTC forecast times.
8. At least 13 METAR observations are required: current + 12 previous visibility observations.

INPUTS
METAR:
  YYYYMMDDHHMM METAR ...
  Prefix is UTC. The embedded Z timestamp should agree with it.

PM2.5 / PBLH:
  YYYY-MM-DD HH:MM, value
  These timestamps are interpreted as IST.

MODEL
24 native LightGBM .txt files are expected in:
mark02_models_metar_pm25_blh\

RUN
1. Put the model folder beside app.py.
2. Run run_app.bat.
3. Paste 13+ METARs.
4. PM2.5 and PBLH can be supplied independently or left blank.
5. Click Generate 12-hour forecast.

TIMEZONE NOTE
The original training script generated hour/minute/month/day-of-year/day-of-week/night/winter directly from the METAR `valid` timestamps without timezone conversion. This app preserves that model-time semantics for compatibility while presenting operational times in IST and aligning the separate IST data sources correctly.
