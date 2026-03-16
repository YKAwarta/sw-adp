"""
EgyptSat-1 Dataset Inspection Script
Run: python inspect_egyptsat.py
Place SatelliteSpaceWeatherData_all.csv in the same directory, or pass path as argument.
"""
import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "SatelliteSpaceWeatherData_all.csv"
df = pd.read_csv(path)

print("=" * 60)
print("EGYPTSAT-1 DATASET INSPECTION")
print("=" * 60)

print(f"\nShape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for i, c in enumerate(df.columns):
    print(f"  {i:3d}. {c}  [{df[c].dtype}]")

print("\n" + "-" * 60)
print("FIRST 5 ROWS:")
print(df.head().to_string())

print("\n" + "-" * 60)
print("SUMMARY STATS:")
print(df.describe().to_string())

print("\n" + "-" * 60)
print("MISSING VALUES:")
missing = df.isnull().sum()
print(missing[missing > 0].to_string() if missing.any() else "None")

print("\n" + "-" * 60)
print("POTENTIAL LABEL COLUMNS:")
label_cols = [c for c in df.columns if any(k in c.lower() for k in ['anomal', 'label', 'target', 'class', 'fault', 'flag'])]
print(label_cols if label_cols else "None found")

print("\n" + "-" * 60)
print("POTENTIAL DATE/TIME COLUMNS:")
date_cols = [c for c in df.columns if any(k in c.lower() for k in ['date', 'time', 'year', 'month', 'day', 'timestamp'])]
print(date_cols if date_cols else "None found")
for dc in date_cols:
    print(f"  {dc}: min={df[dc].min()}, max={df[dc].max()}")

print("\n" + "-" * 60)
print("UNIQUE VALUES PER COLUMN (if < 20):")
for c in df.columns:
    nuniq = df[c].nunique()
    if nuniq < 20:
        print(f"  {c}: {nuniq} unique -> {sorted(df[c].dropna().unique())}")

print("\nDone. Paste this entire output back to Claude.")
