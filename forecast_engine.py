from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import re
import json
from typing import Iterable

import numpy as np
import pandas as pd
import lightgbm as lgb

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "mark02_models_metar_pm25_blh"

UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")

# IMPORTANT: These are the exact 39 features expected by the native models.
# The original training code created its time features directly from the METAR
# `valid` timestamp. Therefore model time features remain in UTC for schema/
# training compatibility. Display and cross-source alignment are done in IST.
FEATURES = (
    [f"vis_lag_{lag}" for lag in range(1, 13)]
    + ["vis_roll_mean_3", "vis_roll_mean_6", "vis_roll_min_6", "vis_roll_std_6"]
    + ["wind", "temp", "dew", "fog", "mist"]
    + ["hour", "minute", "month", "day_of_year", "day_of_week", "night", "winter"]
    + ["pm25", "pm25_lag_1", "pm25_lag_2", "pm25_lag_4", "pm25_roll_mean_4", "pm25_roll_mean_12"]
    + ["inv_height", "inv_strength_k", "pblh_temp_c", "inv_top_temp_c", "inv_2_layer"]
)
EXPECTED_FEATURE_COUNT = 39
N_MODELS = 24
CALIBRATION_PATH = BASE_DIR / "calibration.json"


@dataclass
class ParsedMETAR:
    timestamp_utc: pd.Timestamp
    visibility_m: float
    wind: float
    temp: float
    dew: float
    fog: int
    mist: int
    raw: str

    @property
    def timestamp_ist(self) -> pd.Timestamp:
        return self.timestamp_utc.tz_convert(IST)


def _utc_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(UTC)
    return ts.tz_convert(UTC)


def _ist_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    return _utc_timestamp(value).tz_convert(IST)


def _parse_timestamp_prefix(line: str) -> pd.Timestamp:
    m = re.match(r"^\s*(\d{12})\s+", line)
    if not m:
        raise ValueError(
            "Each METAR must begin with YYYYMMDDHHMM in UTC, e.g. "
            "202609261300 METAR VIDP 261300Z ..."
        )
    naive = pd.to_datetime(m.group(1), format="%Y%m%d%H%M", errors="raise")
    return naive.tz_localize(UTC)


def _parse_embedded_z_time(line: str) -> pd.Timestamp | None:
    m = re.search(r"\b(\d{2})(\d{4})Z\b", line.upper())
    if not m:
        return None
    # The date is the day of month from the embedded METAR group. We cannot
    # reliably infer month/year across month boundaries from the group alone,
    # so this is only used as a consistency check against the prefix.
    return None


def _parse_visibility(tokens: list[str]) -> float:
    for tok in tokens:
        t = tok.upper().strip("=,")
        if re.fullmatch(r"\d{4}", t):
            return float(t)
        if t == "CAVOK":
            return 10000.0
    raise ValueError("Could not parse METAR visibility (expected 4-digit metres or CAVOK).")


def _parse_wind(metar: str) -> float:
    m = re.search(r"\b(?:VRB|(\d{3}))(?P<speed>\d{2,3})(?:G\d{2,3})?KT\b", metar.upper())
    if not m:
        raise ValueError("Could not parse required METAR field: wind (e.g. 03010KT).")
    return float(m.group("speed"))


def _parse_temp_dew(metar: str) -> tuple[float, float]:
    m = re.search(r"\b(M?\d{2})/(M?\d{2})\b", metar.upper())
    if not m:
        raise ValueError("Could not parse required METAR fields: temperature/dew point.")

    def cv(x: str) -> float:
        return -float(x[1:]) if x.startswith("M") else float(x)

    return cv(m.group(1)), cv(m.group(2))


def parse_metar(line: str) -> ParsedMETAR:
    line = line.strip()
    if not line:
        raise ValueError("Blank METAR line.")
    ts = _parse_timestamp_prefix(line)
    upper = line.upper()
    tokens = upper.split()
    visibility = _parse_visibility(tokens)
    wind = _parse_wind(upper)
    temp, dew = _parse_temp_dew(upper)
    fog = int("FG" in upper.split())
    mist = int("BR" in upper.split())
    return ParsedMETAR(ts, visibility, wind, temp, dew, fog, mist, line)


def parse_metar_block(text: str) -> list[ParsedMETAR]:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines:
        raise ValueError("Enter at least one timestamped METAR.")
    rows = [parse_metar(x) for x in lines]
    rows.sort(key=lambda x: x.timestamp_utc)
    dedup = {row.timestamp_utc: row for row in rows}
    return sorted(dedup.values(), key=lambda x: x.timestamp_utc)


def fetch_latest_ogimet_metars(hours: int = 24):
    """Fetch and parse recent VIDP METAR reports from Ogimet."""
    import csv
    import io
    from datetime import datetime, timedelta, timezone

    import requests

    if hours <= 0:
        raise ValueError("hours must be greater than zero.")

    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(hours=hours)

    url = "https://www.ogimet.com/cgi-bin/getmetar"
    params = {
        "icao": "VIDP",
        "begin": start_utc.strftime("%Y%m%d%H%M"),
        "end": now_utc.strftime("%Y%m%d%H%M"),
        "lang": "eng",
        "header": "yes",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 Chrome/130.0 Safari/537.36"
    }

    try:
        response = requests.get(
            url, params=params, headers=headers, timeout=30
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not retrieve METAR data from Ogimet: {exc}") from exc

    body = response.text.strip()
    if not body:
        raise RuntimeError("Ogimet returned an empty response.")

    # Read CSV using the actual header returned by Ogimet.
    reader = csv.DictReader(io.StringIO(body))
    if not reader.fieldnames:
        raise RuntimeError("Ogimet returned data without a CSV header.")

    # Normalize column names to handle accents and whitespace.
    column_map = {
        name.strip().upper(): name
        for name in reader.fieldnames
        if name
    }

    def find_column(*names):
        for name in names:
            if name in column_map:
                return column_map[name]
        return None

    year_col = find_column("YEAR", "ANO", "AÑO")
    month_col = find_column("MONTH", "MES")
    day_col = find_column("DAY", "DIA", "DÍA")
    hour_col = find_column("HOUR", "HORA")
    minute_col = find_column("MIN", "MINUTO")
    report_col = find_column("REPORT", "PARTE", "METAR")

    required = [
        year_col, month_col, day_col,
        hour_col, minute_col, report_col
    ]
    if any(col is None for col in required):
        raise RuntimeError(
            "Unrecognized Ogimet CSV columns: "
            + ", ".join(reader.fieldnames)
        )

    parsed = []
    invalid_reports = 0

    for row in reader:
        try:
            report = str(row.get(report_col, "")).strip()
            if not report:
                continue

            timestamp = datetime(
                int(row[year_col]),
                int(row[month_col]),
                int(row[day_col]),
                int(row[hour_col]),
                int(row[minute_col]),
                tzinfo=timezone.utc,
            )

            if not (start_utc <= timestamp <= now_utc):
                continue

            # Remove a duplicate timestamp prefix if the report already has one.
            report = report.strip()
            line = f"{timestamp.strftime('%Y%m%d%H%M')} {report}"

            try:
                metar = parse_metar(line)
            except (ValueError, TypeError, KeyError):
                # Try the original report if it already contains its own
                # timestamp or if the parser expects a different prefix format.
                try:
                    metar = parse_metar(report)
                except (ValueError, TypeError, KeyError):
                    invalid_reports += 1
                    continue

            parsed.append(metar)

        except (ValueError, TypeError, KeyError):
            invalid_reports += 1

    # Deduplicate and sort by UTC observation time.
    unique = {
        metar.timestamp_utc: metar
        for metar in parsed
    }
    metars = sorted(
        unique.values(),
        key=lambda metar: metar.timestamp_utc,
    )

    if len(metars) < 13:
        preview = body[:300].replace("\n", " ")
        raise RuntimeError(
            f"Only {len(metars)} usable VIDP METAR reports were found "
            f"in the last {hours} hours; at least 13 are required. "
            f"Unparseable reports: {invalid_reports}. "
            f"Ogimet response preview: {preview}"
        )

    return metars



def parse_datetime_value_table(text: str, value_name: str) -> pd.DataFrame:
    """Parse timestamp + numeric value. Input timestamps are interpreted as IST."""
    records = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s*[,;\t]\s*", line)
        if len(parts) >= 2:
            dt_text, value_text = parts[0].strip(), parts[1].strip()
        else:
            m = re.match(r"^(.+?)\s+(-?\d+(?:\.\d+)?)\s*$", line)
            if not m:
                raise ValueError(
                    f"Could not parse {value_name} row: {line!r}. Use: YYYY-MM-DD HH:MM, value"
                )
            dt_text, value_text = m.group(1), m.group(2)
        dt = pd.to_datetime(dt_text, errors="coerce")
        val = pd.to_numeric(value_text, errors="coerce")
        if pd.isna(dt) or pd.isna(val):
            raise ValueError(f"Invalid {value_name} row: {line!r}.")
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            ts = ts.tz_localize(IST)
        else:
            ts = ts.tz_convert(IST)
        records.append((ts, float(val)))

    if not records:
        # Optional sources are allowed to be empty.
        return pd.DataFrame(columns=["timestamp", value_name])

    df = pd.DataFrame(records, columns=["timestamp", value_name])
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _asof_value_or_nan(history: pd.DataFrame, origin_ist: pd.Timestamp, value_col: str) -> float:
    if history is None or history.empty or value_col not in history.columns:
        return np.nan
    h = history.sort_values("timestamp")
    eligible = h[h["timestamp"] <= origin_ist]
    if eligible.empty:
        return np.nan
    value = eligible.iloc[-1][value_col]
    return float(value) if pd.notna(value) else np.nan


def _asof_pm_series(pm25_df: pd.DataFrame, metar_times_ist: list[pd.Timestamp]) -> np.ndarray:
    """Align PM2.5 independently to METAR times in IST. Missing periods remain NaN."""
    if pm25_df is None or pm25_df.empty:
        return np.full(len(metar_times_ist), np.nan, dtype=float)
    h = pm25_df.sort_values("timestamp")
    out = []
    for t in metar_times_ist:
        eligible = h[h["timestamp"] <= t]
        if eligible.empty:
            out.append(np.nan)
        else:
            v = eligible.iloc[-1]["pm25"]
            out.append(float(v) if pd.notna(v) else np.nan)
    return np.asarray(out, dtype=float)


def _lag_or_nan(arr: np.ndarray, offset: int) -> float:
    idx = len(arr) - 1 - offset
    if idx < 0:
        return np.nan
    return float(arr[idx])


def _mean_ignore_nan(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    return float(np.mean(vals)) if len(vals) else np.nan


def build_feature_row(
    metars: list[ParsedMETAR],
    pm25_df: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    pblh_height_df: pd.DataFrame,
    inv_strength_df: pd.DataFrame,
    pblh_temp_df: pd.DataFrame,
    inv_top_temp_df: pd.DataFrame,
    inv_2_layer_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    if len(metars) < 13:
        raise ValueError(
            f"At least 13 timestamped METARs are required: current + 12 previous observations. You supplied {len(metars)}."
        )

    mets = sorted(metars, key=lambda x: x.timestamp_utc)
    origin = _utc_timestamp(forecast_origin)
    if mets[-1].timestamp_utc != origin:
        raise ValueError("The latest METAR must define the forecast origin.")

    previous = np.array([m.visibility_m for m in mets[:-1]], dtype=float)[-12:]
    lag_values = list(previous[::-1])
    latest = mets[-1]
    row: dict[str, float] = {}

    for i, value in enumerate(lag_values, 1):
        row[f"vis_lag_{i}"] = float(value)
    row["vis_roll_mean_3"] = float(np.mean(previous[-3:]))
    row["vis_roll_mean_6"] = float(np.mean(previous[-6:]))
    row["vis_roll_min_6"] = float(np.min(previous[-6:]))
    row["vis_roll_std_6"] = float(np.std(previous[-6:], ddof=1))

    row.update({
        "wind": latest.wind,
        "temp": latest.temp,
        "dew": latest.dew,
        "fog": latest.fog,
        "mist": latest.mist,
    })

    # Preserve the exact time-feature semantics of the trained models: UTC.
    t = origin
    row.update({
        "hour": t.hour,
        "minute": t.minute,
        "month": t.month,
        "day_of_year": t.dayofyear,
        "day_of_week": t.dayofweek,
        "night": int(t.hour < 6 or t.hour >= 18),
        "winter": int(t.month in (11, 12, 1, 2)),
    })

    # Optional PM2.5 is independent. We align it in IST, and leave unavailable
    # values as NaN. LightGBM natively supports missing numeric values.
    met_times_ist = [m.timestamp_ist for m in mets]
    pm = _asof_pm_series(pm25_df, met_times_ist)
    row["pm25"] = _lag_or_nan(pm, 0)
    row["pm25_lag_1"] = _lag_or_nan(pm, 1)
    row["pm25_lag_2"] = _lag_or_nan(pm, 2)
    row["pm25_lag_4"] = _lag_or_nan(pm, 4)
    row["pm25_roll_mean_4"] = _mean_ignore_nan(pm[-5:-1])
    row["pm25_roll_mean_12"] = _mean_ignore_nan(pm[-13:-1])

    origin_ist = origin.tz_convert(IST)
    pblh_inputs = [
        ("inv_height", pblh_height_df),
        ("inv_strength_k", inv_strength_df),
        ("pblh_temp_c", pblh_temp_df),
        ("inv_top_temp_c", inv_top_temp_df),
        ("inv_2_layer", inv_2_layer_df),
    ]
    for name, df in pblh_inputs:
        row[name] = _asof_value_or_nan(df, origin_ist, name)

    X = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES, dtype=float)
    source_status = {
        "metar": True,
        "pm25_current_available": bool(np.isfinite(row["pm25"])),
        "pblh_available": all(np.isfinite(row[n]) for n, _ in pblh_inputs),
        "pm25_missing_features": int(sum(not np.isfinite(row[n]) for n in FEATURES if n.startswith("pm25"))),
        "origin_utc": origin,
        "origin_ist": origin_ist,
    }
    return X, source_status


def load_models() -> list[lgb.Booster]:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
    models = []
    for i in range(1, N_MODELS + 1):
        path = MODEL_DIR / f"model_{i:02d}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing model: {path}")
        model = lgb.Booster(model_file=str(path))
        names = list(model.feature_name())
        if len(names) != EXPECTED_FEATURE_COUNT or names != FEATURES:
            raise ValueError(f"{path.name} feature schema mismatch. Expected exact Mark-2 39-feature order.")
        models.append(model)
    return models


def load_calibration() -> dict:
    if not CALIBRATION_PATH.exists():
        return {}
    try:
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Never let a broken optional calibration file stop inference.
        return {}


def _apply_calibration(pred: float, spec: dict) -> float:
    if not spec:
        return max(0.0, pred)
    method = spec.get("method", "none")
    if method == "bias":
        value = pred + float(spec.get("delta", 0.0))
    elif method == "affine":
        value = float(spec.get("a", 0.0)) + float(spec.get("b", 1.0)) * pred
    elif method == "log_affine":
        value = np.expm1(float(spec.get("a", 0.0)) + float(spec.get("b", 1.0)) * np.log1p(max(pred, 0.0)))
    elif method == "isotonic":
        xs = np.asarray(spec.get("x", []), dtype=float)
        ys = np.asarray(spec.get("y", []), dtype=float)
        value = float(np.interp(pred, xs, ys)) if len(xs) >= 2 else pred
    else:
        value = pred
    return max(0.0, float(value))


def round_visibility(value: float) -> int:
    """Operational display rounding requested for VIDP forecasts."""
    if not np.isfinite(value):
        return 0
    step = 100 if value > 800 else 50
    return int(np.floor(value / step + 0.5) * step)


def forecast_24_steps(models: list[lgb.Booster], X: pd.DataFrame, origin_utc: pd.Timestamp) -> pd.DataFrame:
    values = []
    origin_utc = _utc_timestamp(origin_utc)
    calibration = load_calibration()
    model_specs = calibration.get("models", {})
    for step, model in enumerate(models, 1):
        raw = max(0.0, float(model.predict(X)[0]))
        spec = model_specs.get(str(step), {"method": "none"})
        calibrated = _apply_calibration(raw, spec)
        rounded = round_visibility(calibrated)
        values.append({
            "step": step,
            "forecast_time_utc": origin_utc + pd.Timedelta(minutes=30 * step),
            "forecast_time": (origin_utc + pd.Timedelta(minutes=30 * step)).tz_convert(IST),
            "minutes_ahead": 30 * step,
            "visibility_model_m": raw,
            "visibility_calibrated_m": calibrated,
            "visibility_m": rounded,
            "visibility_km": rounded / 1000.0,
            "calibration_method": spec.get("method", "none"),
        })
    return pd.DataFrame(values)


def run_forecast(
    metars: list[ParsedMETAR],
    pm25_df: pd.DataFrame,
    pblh_height_df: pd.DataFrame,
    inv_strength_df: pd.DataFrame,
    pblh_temp_df: pd.DataFrame,
    inv_top_temp_df: pd.DataFrame,
    inv_2_layer_df: pd.DataFrame,
    models: list[lgb.Booster] | None = None,
):
    if models is None:
        models = load_models()
    origin = max(m.timestamp_utc for m in metars)
    X, source_status = build_feature_row(
        metars, pm25_df, origin,
        pblh_height_df, inv_strength_df, pblh_temp_df, inv_top_temp_df, inv_2_layer_df,
    )
    forecast = forecast_24_steps(models, X, origin)
    return forecast, X, source_status
