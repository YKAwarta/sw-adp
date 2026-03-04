"""
SW-ADP: Verify All Downloaded Datasets
Loads OPSSAT-AD and space weather data, prints comprehensive stats,
and checks alignment feasibility.

Usage:
    conda activate sw-adp
    python scripts/verify_data.py
"""

import sys
from pathlib import Path

def verify_opssat():
    """Load and inspect OPSSAT-AD datasets."""
    import pandas as pd
    import numpy as np

    print("=" * 60)
    print("OPSSAT-AD DATASET")
    print("=" * 60)

    opssat_dir = Path("data") / "opssat"

    # --- segments.csv (raw telemetry) ---
    seg_path = opssat_dir / "segments.csv"
    if not seg_path.exists():
        print(f"  [MISSING] {seg_path}")
        return False

    df_seg = pd.read_csv(seg_path)
    print(f"\nsegments.csv (raw telemetry):")
    print(f"  Shape: {df_seg.shape}")
    print(f"  Columns: {list(df_seg.columns)}")
    print(f"  Memory: {df_seg.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Check for label/channel columns
    if "label" in df_seg.columns:
        labels = df_seg["label"].value_counts()
        print(f"  Labels:")
        for lab, cnt in labels.items():
            print(f"    {lab}: {cnt:,} ({cnt/len(df_seg)*100:.1f}%)")

    if "channel" in df_seg.columns:
        channels = df_seg["channel"].unique()
        print(f"  Channels ({len(channels)}): {sorted(channels)}")

    # Numeric summary
    num_cols = df_seg.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        print(f"  Numeric columns: {list(num_cols)}")
        print(f"  NaN counts:")
        nans = df_seg[num_cols].isna().sum()
        for col, n in nans.items():
            if n > 0:
                print(f"    {col}: {n:,}")
        if nans.sum() == 0:
            print(f"    (none)")

    # --- dataset.csv (extracted features) ---
    ds_path = opssat_dir / "dataset.csv"
    if not ds_path.exists():
        print(f"\n  [MISSING] {ds_path}")
        return False

    df_ds = pd.read_csv(ds_path)
    print(f"\ndataset.csv (extracted features):")
    print(f"  Shape: {df_ds.shape}")
    print(f"  Columns ({len(df_ds.columns)}): {list(df_ds.columns[:10])}{'...' if len(df_ds.columns) > 10 else ''}")

    if "label" in df_ds.columns:
        labels = df_ds["label"].value_counts()
        print(f"  Labels:")
        for lab, cnt in labels.items():
            print(f"    {lab}: {cnt:,} ({cnt/len(df_ds)*100:.1f}%)")

    if "split" in df_ds.columns:
        splits = df_ds["split"].value_counts()
        print(f"  Train/test split:")
        for s, cnt in splits.items():
            print(f"    {s}: {cnt:,}")

    return True


def verify_space_weather():
    """Load and inspect space weather datasets."""
    import pandas as pd

    print("\n" + "=" * 60)
    print("SPACE WEATHER INDICES")
    print("=" * 60)

    sw_dir = Path("data") / "space_weather"

    # --- CelesTrak daily data ---
    kp_path = sw_dir / "kp_f107_daily.csv"
    if kp_path.exists():
        df_kp = pd.read_csv(kp_path, parse_dates=["date"])
        print(f"\nkp_f107_daily.csv (CelesTrak):")
        print(f"  Shape: {df_kp.shape}")
        print(f"  Date range: {df_kp['date'].min()} to {df_kp['date'].max()}")
        print(f"  Columns: {list(df_kp.columns)}")
        if "kp_avg" in df_kp.columns:
            print(f"  Kp avg: min={df_kp['kp_avg'].min():.1f}, max={df_kp['kp_avg'].max():.1f}, mean={df_kp['kp_avg'].mean():.2f}")
        if "f10_7" in df_kp.columns:
            print(f"  F10.7: min={df_kp['f10_7'].min():.1f}, max={df_kp['f10_7'].max():.1f}, mean={df_kp['f10_7'].mean():.1f}")
        nans = df_kp.isna().sum()
        if nans.sum() > 0:
            print(f"  Missing values: {nans[nans > 0].to_dict()}")
    else:
        print(f"\n  [MISSING] {kp_path}")

    # --- OMNI2 hourly data ---
    omni_path = sw_dir / "omni_hourly.csv"
    if omni_path.exists():
        df_omni = pd.read_csv(omni_path, parse_dates=["datetime"])
        print(f"\nomni_hourly.csv (NASA OMNI2):")
        print(f"  Shape: {df_omni.shape}")
        print(f"  Date range: {df_omni['datetime'].min()} to {df_omni['datetime'].max()}")
        print(f"  Columns: {list(df_omni.columns)}")

        for col in ["dst", "kp", "f10_7", "sw_speed", "sw_proton_density", "imf_bz_gsm"]:
            if col in df_omni.columns:
                valid = df_omni[col].dropna()
                print(f"  {col:20s}: min={valid.min():>8.1f}, max={valid.max():>8.1f}, mean={valid.mean():>8.2f}, missing={df_omni[col].isna().sum():,}")

        # Check for storm periods (Dst < -50 or Kp >= 5)
        storms_dst = df_omni[df_omni["dst"] < -50] if "dst" in df_omni.columns else pd.DataFrame()
        storms_kp = df_omni[df_omni["kp"] >= 5.0] if "kp" in df_omni.columns else pd.DataFrame()
        print(f"\n  Storm periods (useful for ablation):")
        print(f"    Hours with Dst < -50 nT: {len(storms_dst):,}")
        print(f"    Hours with Kp >= 5.0:    {len(storms_kp):,}")
    else:
        print(f"\n  [MISSING] {omni_path}")

    return True


def check_alignment():
    """Check if OPSSAT and space weather data can be aligned."""
    import pandas as pd

    print("\n" + "=" * 60)
    print("DATA ALIGNMENT CHECK")
    print("=" * 60)

    seg_path = Path("data") / "opssat" / "segments.csv"
    omni_path = Path("data") / "space_weather" / "omni_hourly.csv"

    if not seg_path.exists() or not omni_path.exists():
        print("  [SKIP] Cannot check alignment; files missing.")
        return

    df_seg = pd.read_csv(seg_path)
    df_omni = pd.read_csv(omni_path, parse_dates=["datetime"])

    # Check if segments have timestamp info
    time_cols = [c for c in df_seg.columns if "time" in c.lower() or "date" in c.lower() or "stamp" in c.lower()]
    if time_cols:
        print(f"  OPSSAT time columns found: {time_cols}")
        print(f"  OMNI date range: {df_omni['datetime'].min()} to {df_omni['datetime'].max()}")
        print(f"  Alignment feasibility: LIKELY POSSIBLE")
        print(f"  Note: OPSSAT is Hz-frequency, OMNI is hourly.")
        print(f"  Strategy: forward-fill or interpolate OMNI to match OPSSAT timestamps.")
    else:
        print(f"  OPSSAT columns: {list(df_seg.columns)}")
        print(f"  No obvious timestamp column found in segments.csv.")
        print(f"  You may need to use segment IDs + metadata to map to dates.")
        print(f"  Check the OPSSAT-AD paper for the telemetry collection dates")
        print(f"  (May 2024, during final OPS-SAT mission operations).")


def main():
    print()
    print("*" * 60)
    print("  SW-ADP DATA VERIFICATION")
    print("*" * 60)

    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("[FAIL] pandas/numpy not installed. Run: conda activate sw-adp")
        return 1

    opssat_ok = verify_opssat()
    sw_ok = verify_space_weather()
    check_alignment()

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"  OPSSAT-AD:      {'OK' if opssat_ok else 'ISSUES'}")
    print(f"  Space weather:  {'OK' if sw_ok else 'ISSUES'}")
    print()
    print("If both OK, you're ready to build the data pipeline.")
    print("Next step: EDA notebook to explore channel-SW correlations.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
