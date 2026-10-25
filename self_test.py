from pathlib import Path
import lightgbm as lgb

from forecast_engine import BASE_DIR, MODEL_DIR, FEATURES, EXPECTED_FEATURE_COUNT, N_MODELS, CALIBRATION_PATH, round_visibility

print("Python/project:", BASE_DIR)
print("Model directory:", MODEL_DIR)
print("Features expected:", EXPECTED_FEATURE_COUNT)
print("Calibration file:", CALIBRATION_PATH if CALIBRATION_PATH.exists() else "not present (raw model mode)")
print("Rounding check:", [(x, round_visibility(x)) for x in (801, 850, 899, 800, 799, 225)])

if not MODEL_DIR.exists():
    raise SystemExit(f"FAIL: model directory does not exist: {MODEL_DIR}")

for i in range(1, N_MODELS + 1):
    path = MODEL_DIR / f"model_{i:02d}.txt"
    if not path.exists():
        raise SystemExit(f"FAIL: missing {path.name}")

print(f"OK: all {N_MODELS} model files found.")

for i in range(1, N_MODELS + 1):
    path = MODEL_DIR / f"model_{i:02d}.txt"
    model = lgb.Booster(model_file=str(path))
    names = list(model.feature_name())
    if len(names) != EXPECTED_FEATURE_COUNT:
        raise SystemExit(f"FAIL: {path.name} has {len(names)} features")
    if names != FEATURES:
        raise SystemExit(f"FAIL: {path.name} feature order/schema does not match")
    print(f"OK: {path.name} | {len(names)} features | schema matches")

print("=" * 60)
print("PASS: all 24 native LightGBM models are valid and compatible.")
print("=" * 60)
