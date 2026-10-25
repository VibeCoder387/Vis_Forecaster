VIDP MARK-2 OPTIMIZATION
========================

1) Rounding is now built into forecast_engine.py:
   - value > 800 m  -> nearest 100 m
   - value <= 800 m -> nearest 50 m
   The raw LightGBM prediction and calibrated prediction remain visible in the table.

2) Calibration is data-driven and horizon-specific.
   Run:

       python optimize_pipeline.py

   It expects:

       mark02_models_metar_pm25_blh\processed_training_data.csv

   The original Mark-2 training script created this CSV. The optimizer:
   - keeps the existing 24 native models unchanged;
   - uses 80-90% of the chronological data to fit calibration;
   - uses 90-95% to select among no correction, additive bias, affine,
     log-affine, and monotonic isotonic calibration;
   - uses the final 5% only as an audit;
   - refuses a selected correction if it materially worsens final MAE;
   - writes calibration.json.

3) The Streamlit app automatically uses calibration.json when present.
   If it is absent, the app uses the native model predictions unchanged.

IMPORTANT:
Do not add a fixed +500 m / +20% rule. The model should only be shifted if the
historical calibration data demonstrates a systematic bias. This keeps the
pipeline defensible for the IMD project.

The optimizer is intentionally separate from the live app. The live app never
re-trains models.
