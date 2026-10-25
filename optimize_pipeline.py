
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "mark02_models_metar_pm25_blh"
DATA_PATH = MODEL_DIR / "processed_training_data.csv"
CALIBRATION_PATH = BASE_DIR / "calibration.json"

FEATURES = (
    [f"vis_lag_{i}" for i in range(1, 13)]
    + ["vis_roll_mean_3", "vis_roll_mean_6", "vis_roll_min_6", "vis_roll_std_6"]
    + ["wind", "temp", "dew", "fog", "mist"]
    + ["hour", "minute", "month", "day_of_year", "day_of_week", "night", "winter"]
    + ["pm25", "pm25_lag_1", "pm25_lag_2", "pm25_lag_4",
       "pm25_roll_mean_4", "pm25_roll_mean_12"]
    + ["inv_height", "inv_strength_k", "pblh_temp_c",
       "inv_top_temp_c", "inv_2_layer"]
)


def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "bias": float(np.mean(p - y)),
    }


def fit_bias(x, y):
    return {"method": "bias", "delta": float(np.mean(y - x))}


def fit_affine(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "method": "affine",
        "a": float(np.clip(intercept, -2500, 2500)),
        "b": float(np.clip(slope, 0.5, 1.5)),
    }


def fit_log_affine(x, y):
    lx = np.log1p(np.maximum(x, 0))
    ly = np.log1p(np.maximum(y, 0))
    slope, intercept = np.polyfit(lx, ly, 1)
    return {
        "method": "log_affine",
        "a": float(np.clip(intercept, -1.5, 1.5)),
        "b": float(np.clip(slope, 0.5, 1.5)),
    }


def fit_isotonic(x, y):
    order = np.argsort(x, kind="mergesort")
    xs = np.asarray(x, dtype=float)[order]
    ys = np.asarray(y, dtype=float)[order]

    # Pool-adjacent-violators algorithm.
    blocks = []
    for xv, yv in zip(xs, ys):
        blocks.append([xv, xv, 1, yv])
        while (
            len(blocks) >= 2
            and blocks[-2][3] / blocks[-2][2]
            > blocks[-1][3] / blocks[-1][2]
        ):
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([
                left[0], right[1],
                left[2] + right[2],
                left[3] + right[3],
            ])

    out_x, out_y = [], []
    for left, _, count, total in blocks:
        out_x.append(float(left))
        out_y.append(float(total / count))

    # np.interp needs increasing x values.
    unique_x, inverse = np.unique(out_x, return_inverse=True)
    if len(unique_x) != len(out_x):
        sums = np.zeros(len(unique_x))
        counts = np.zeros(len(unique_x))
        for i, j in enumerate(inverse):
            sums[j] += out_y[i]
            counts[j] += 1
        out_x = unique_x.tolist()
        out_y = (sums / counts).tolist()

    return {"method": "isotonic", "x": out_x, "y": out_y}


def apply_calibration(pred, spec):
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


def evaluate_visibility(y, raw, calibrated):
    result = {}
    for threshold in (1000, 2000):
        mask = y < threshold
        result[f"below_{threshold}m"] = {
            "n": int(mask.sum()),
            "raw_mae": (
                float(np.mean(np.abs(y[mask] - raw[mask])))
                if mask.any() else None
            ),
            "cal_mae": (
                float(np.mean(np.abs(y[mask] - calibrated[mask])))
                if mask.any() else None
            ),
        }
    return result


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing processed data: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns: {missing}")

    targets = [f"target_{i}" for i in range(1, 25)]
    missing_targets = [c for c in targets if c not in df.columns]
    if missing_targets:
        raise SystemExit(f"Missing target columns: {missing_targets}")

    # Remove rows without real timestamps. Do not discard rows just because
    # some features are missing: native LightGBM supports missing features.
    df["valid"] = pd.to_datetime(df["valid"], errors="coerce")
    df = (
        df.dropna(subset=["valid"])
        .sort_values("valid")
        .reset_index(drop=True)
    )

    if df["valid"].duplicated().any():
        raise SystemExit(
            "Duplicate real timestamps detected. Resolve them before calibration."
        )

    n = len(df)
    i80, i90, i95 = int(n * 0.80), int(n * 0.90), int(n * 0.95)

    fit_df = df.iloc[i80:i90]
    select_df = df.iloc[i90:i95]
    audit_df = df.iloc[i95:]

    print(f"Rows with valid timestamps: {n:,}")
    print(f"Calibration fit: {len(fit_df):,}")
    print(f"Method selection: {len(select_df):,}")
    print(f"Final audit: {len(audit_df):,}")
    print("Missing feature values are retained for LightGBM.")

    calibration = {
        "version": 2,
        "description": "Horizon-specific post-hoc calibration.",
        "selection_scheme": (
            "Chronological 80-90% fit, 90-95% selection, "
            "95-100% audit. Audit does not choose or reject calibration."
        ),
        "models": {},
    }
    results = []

    for step in range(1, 25):
        model_path = MODEL_DIR / f"model_{step:02d}.txt"
        if not model_path.exists():
            raise SystemExit(f"Missing model: {model_path}")

        model = lgb.Booster(model_file=str(model_path))
        target = f"target_{step}"

        def predict(part):
            # Keep rows with missing features; drop only missing targets.
            usable = part.dropna(subset=[target])
            y = usable[target].to_numpy(dtype=float)
            p = model.predict(usable[FEATURES])
            valid = np.isfinite(y) & np.isfinite(p)
            return y[valid], np.asarray(p)[valid]

        y_fit, p_fit = predict(fit_df)
        y_sel, p_sel = predict(select_df)
        y_audit, p_audit = predict(audit_df)

        if len(y_fit) < 100 or len(y_sel) < 100 or len(y_audit) < 100:
            raise SystemExit(
                f"Too few usable rows for horizon {step}: "
                f"fit={len(y_fit)}, select={len(y_sel)}, audit={len(y_audit)}"
            )

        candidates = {"none": {"method": "none"}}

        candidates["bias"] = fit_bias(p_fit, y_fit)
        candidates["affine"] = fit_affine(p_fit, y_fit)
        candidates["log_affine"] = fit_log_affine(p_fit, y_fit)

        # Isotonic calibration can overfit, so keep it as a candidate and
        # let the separate selection period decide whether it is useful.
        candidates["isotonic"] = fit_isotonic(p_fit, y_fit)

        selection_scores = {}
        for name, spec in candidates.items():
            pred = apply_calibration(p_sel, spec)
            selection_scores[name] = metrics(y_sel, pred)

        # Pick using selection data only. Never use audit performance here.
        chosen = min(
            selection_scores,
            key=lambda name: (
                selection_scores[name]["mae"],
                abs(selection_scores[name]["bias"]),
            ),
        )
        spec = candidates[chosen]

        raw = metrics(y_audit, p_audit)
        calibrated_pred = apply_calibration(p_audit, spec)
        calibrated = metrics(y_audit, calibrated_pred)

        low_visibility = evaluate_visibility(
            y_audit, p_audit, calibrated_pred
        )

        calibration["models"][str(step)] = spec
        results.append({
            "step": step,
            "minutes": step * 30,
            "chosen": chosen,
            "raw_mae": raw["mae"],
            "cal_mae": calibrated["mae"],
            "raw_bias": raw["bias"],
            "cal_bias": calibrated["bias"],
            "audit_rows": raw["n"],
            "raw_lowvis_1000": low_visibility["below_1000m"]["raw_mae"],
            "cal_lowvis_1000": low_visibility["below_1000m"]["cal_mae"],
            "raw_lowvis_2000": low_visibility["below_2000m"]["raw_mae"],
            "cal_lowvis_2000": low_visibility["below_2000m"]["cal_mae"],
        })

    # Write only after all 24 horizons finish successfully.
    temp_path = CALIBRATION_PATH.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    temp_path.replace(CALIBRATION_PATH)

    out = pd.DataFrame(results)
    print("\nFINAL AUDIT RESULTS (audit did not influence selection)")
    print(out.to_string(index=False, formatters={
        c: (lambda x: "N/A" if pd.isna(x) else f"{x:.1f}")
        for c in [
            "raw_mae", "cal_mae", "raw_bias", "cal_bias",
            "raw_lowvis_1000", "cal_lowvis_1000",
            "raw_lowvis_2000", "cal_lowvis_2000",
        ]
    }))

    print(f"\nSaved calibration: {CALIBRATION_PATH}")
    print("Original LightGBM model files were not modified.")
    print(
        "\nIMPORTANT: Audit metrics are only truly out-of-sample if these "
        "dates were not used to train the native models."
    )


if __name__ == "__main__":
    main()
