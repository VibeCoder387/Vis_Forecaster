import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "mark02_models_metar_pm25_blh"
DATA_PATH = MODEL_DIR / "processed_training_data.csv"
CAL_PATH = ROOT / "calibration.json"
OUT_PATH = ROOT / "mark2_evaluation_diagnostic.csv"

FEATURES = [
    "vis_lag_1", "vis_lag_2", "vis_lag_3", "vis_lag_4",
    "vis_lag_5", "vis_lag_6", "vis_lag_7", "vis_lag_8",
    "vis_lag_9", "vis_lag_10", "vis_lag_11", "vis_lag_12",
    "vis_roll_mean_3", "vis_roll_mean_6", "vis_roll_min_6",
    "vis_roll_std_6", "wind", "temp", "dew", "fog", "mist",
    "hour", "minute", "month", "day_of_year", "day_of_week",
    "night", "winter", "pm25", "pm25_lag_1", "pm25_lag_2",
    "pm25_lag_4", "pm25_roll_mean_4", "pm25_roll_mean_12",
    "inv_height", "inv_strength_k", "pblh_temp_c",
    "inv_top_temp_c", "inv_2_layer",
]

def calibrate(pred, spec):
    pred = np.asarray(pred, dtype=float)
    method = spec["method"]

    if method == "none":
        result = pred
    elif method == "bias":
        result = pred + spec["delta"]
    elif method == "affine":
        result = spec["a"] + spec["b"] * pred
    elif method == "log_affine":
        result = np.expm1(
            spec["a"] + spec["b"] * np.log1p(np.maximum(pred, 0))
        )
    elif method == "isotonic":
        result = np.interp(
            pred, np.asarray(spec["x"]), np.asarray(spec["y"])
        )
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    return np.maximum(result, 0.0)

def mae(y, p):
    return float(np.mean(np.abs(y - p))) if len(y) else np.nan

def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else np.nan

def event_metrics(y, p, threshold):
    actual = y < threshold
    predicted = p < threshold
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    return precision, recall, tp, fp, fn

def main():
    for path in (DATA_PATH, CAL_PATH):
        if not path.exists():
            raise SystemExit(f"Missing required file: {path}")

    df = pd.read_csv(DATA_PATH)
    cal = json.loads(CAL_PATH.read_text(encoding="utf-8"))

    missing = [c for c in FEATURES if c not in df.columns]
    missing += [
        f"target_{i}" for i in range(1, 25)
        if f"target_{i}" not in df.columns
    ]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    df["valid"] = pd.to_datetime(df["valid"], errors="coerce")
    df = (
        df.dropna(subset=["valid"])
        .sort_values("valid")
        .reset_index(drop=True)
    )
    if df["valid"].duplicated().any():
        raise SystemExit("Duplicate timestamps detected; stopping.")

    # Same row-based final 5% used by the existing calibration pipeline.
    # Diagnostic only: these models used the last 20% for early stopping.
    start = int(len(df) * 0.95)
    test = df.iloc[start:].copy()
    print(f"Total rows: {len(df):,}")
    print(f"Diagnostic rows (final 5%): {len(test):,}")
    print(f"Diagnostic period: {test['valid'].min()} to {test['valid'].max()}")
    print("WARNING: not an independent test; model early stopping used the final 20%.")

    X = test[FEATURES]
    rows = []

    for h in range(1, 25):
        model_path = MODEL_DIR / f"model_{h:02d}.txt"
        if not model_path.exists():
            raise SystemExit(f"Missing model: {model_path}")

        y_all = pd.to_numeric(test[f"target_{h}"], errors="coerce").to_numpy()
        raw_all = lgb.Booster(model_file=str(model_path)).predict(X)
        spec = cal.get("horizons", {}).get(str(h))
        if spec is None:
            raise SystemExit(f"Missing calibration entry for horizon {h}")
        cal_all = calibrate(raw_all, spec)

        valid = np.isfinite(y_all) & np.isfinite(raw_all) & np.isfinite(cal_all)
        y, raw, pred = y_all[valid], raw_all[valid], cal_all[valid]

        # Persistence baseline: latest recorded visibility available to the model.
        # vis_lag_1 is the previous 30-minute grid value in this feature setup.
        persist_all = pd.to_numeric(
            test.loc[valid, "vis_lag_1"], errors="coerce"
        ).to_numpy()
        persist_ok = np.isfinite(persist_all)

        low1 = y < 1000
        low2 = y < 2000
        p1000, r1000, tp1000, fp1000, fn1000 = event_metrics(y, pred, 1000)
        p2000, r2000, tp2000, fp2000, fn2000 = event_metrics(y, pred, 2000)

        rows.append({
            "horizon_hours": h * 0.5,
            "n": len(y),
            "calibration_method": spec["method"],
            "raw_MAE_m": mae(y, raw),
            "calibrated_MAE_m": mae(y, pred),
            "raw_RMSE_m": rmse(y, raw),
            "calibrated_RMSE_m": rmse(y, pred),
            "raw_bias_m": float(np.mean(raw - y)) if len(y) else np.nan,
            "calibrated_bias_m": float(np.mean(pred - y)) if len(y) else np.nan,
            "actual_below_1000_n": int(low1.sum()),
            "raw_MAE_actual_below_1000_m": mae(y[low1], raw[low1]),
            "cal_MAE_actual_below_1000_m": mae(y[low1], pred[low1]),
            "actual_below_2000_n": int(low2.sum()),
            "raw_MAE_actual_below_2000_m": mae(y[low2], raw[low2]),
            "cal_MAE_actual_below_2000_m": mae(y[low2], pred[low2]),
            "event1000_precision": p1000,
            "event1000_recall": r1000,
            "event1000_TP": tp1000,
            "event1000_FP": fp1000,
            "event1000_FN": fn1000,
            "event2000_precision": p2000,
            "event2000_recall": r2000,
            "event2000_TP": tp2000,
            "event2000_FP": fp2000,
            "event2000_FN": fn2000,
            "persistence_n": int(persist_ok.sum()),
            "persistence_MAE_m": mae(y[persist_ok], persist_all[persist_ok]),
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUT_PATH, index=False)

    cols = [
        "horizon_hours", "n", "calibration_method",
        "raw_MAE_m", "calibrated_MAE_m",
        "cal_MAE_actual_below_1000_m", "actual_below_1000_n",
        "event1000_precision", "event1000_recall",
        "persistence_MAE_m",
    ]
    print("\nRESULTS (distances in metres; event precision/recall range 0–1):")
    print(result[cols].round(3).to_string(index=False))
    print(f"\nFull results saved to: {OUT_PATH}")
    print("No models or calibration files were modified.")

if __name__ == "__main__":
    main()
