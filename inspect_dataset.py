import pandas as pd
from pathlib import Path

p = Path("mark02_models_metar_pm25_blh/processed_training_data.csv")
df = pd.read_csv(p, usecols=["valid", "target_1", "target_24"])

df["valid"] = pd.to_datetime(df["valid"], errors="coerce")
df = df.dropna(subset=["valid"]).sort_values("valid").reset_index(drop=True)

n = len(df)
i80, i90, i95 = int(n * .80), int(n * .90), int(n * .95)

print(f"Rows: {n:,}")
print(f"First timestamp: {df['valid'].iloc[0]}")
print(f"Last timestamp:  {df['valid'].iloc[-1]}")
print("\nCalibration boundaries:")
for name, i in [("80%", i80), ("90%", i90), ("95%", i95)]:
    print(f"{name}: {df['valid'].iloc[i]}")

print("\nTarget availability by period:")
for name, part in [
    ("First 80%", df.iloc[:i80]),
    ("80–90%", df.iloc[i80:i90]),
    ("90–95%", df.iloc[i90:i95]),
    ("95–100%", df.iloc[i95:])
]:
    print(
        f"{name:10} rows={len(part):7,} "
        f"target_1={part['target_1'].notna().sum():7,} "
        f"target_24={part['target_24'].notna().sum():7,}"
    )

print("\nLargest timestamp gaps:")
gaps = df["valid"].diff().dt.total_seconds().div(60)
print(gaps.nlargest(5).to_string(index=False), "minutes")
