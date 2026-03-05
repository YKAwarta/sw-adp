"""
Step 2: Clean and prepare NOAA space weather data.

Produces a single clean hourly DataFrame covering the OPSSAT period,
with forward-fill for discrete indices and linear interpolation for
continuous measurements (D-06).
"""
import pandas as pd
import numpy as np
from . import config


def load_omni_hourly() -> pd.DataFrame:
    """Load OMNI2 hourly data, filter to OPSSAT period, handle missing values."""
    df = pd.read_csv(config.OMNI_HOURLY_CSV)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()

    # Filter to OPSSAT period with a small buffer for edge alignment
    df = df.loc[config.OPSSAT_START:config.OPSSAT_END]

    n_before = len(df)
    missing_before = df.isna().sum()

    # D-06: Linear interpolation for continuous measurements
    for col in config.SW_INTERP_COLS:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit_direction="both")

    # D-06: Forward-fill for discrete indices (they hold until next update)
    for col in config.SW_FFILL_COLS:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()  # bfill for any leading NaNs

    missing_after = df.isna().sum()

    print(f"  OMNI hourly: {n_before} hours for OPSSAT period")
    print(f"  Missing before: {missing_before.to_dict()}")
    print(f"  Missing after:  {missing_after.to_dict()}")

    return df


def load_celestrak_daily() -> pd.DataFrame:
    """Load CelesTrak daily data, filter to observed-only rows (Q-06)."""
    df = pd.read_csv(config.KP_F107_DAILY_CSV)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()

    n_before = len(df)
    # Remove forecast rows: they have NaN in kp columns
    df = df.dropna(subset=["kp_avg"])
    n_after = len(df)

    # Filter to OPSSAT period
    df = df.loc[config.OPSSAT_START:config.OPSSAT_END]

    print(f"  CelesTrak: {n_before} -> {n_after} rows after removing forecast rows "
          f"({n_before - n_after} removed)")
    print(f"  {len(df)} rows in OPSSAT period")

    return df


def build_sw_hourly() -> pd.DataFrame:
    """
    Build the clean hourly space weather DataFrame for the OPSSAT period.

    Uses OMNI2 as the primary source (hourly Kp, Dst, solar wind, IMF Bz).
    CelesTrak daily provides F10.7 backup if needed, but OMNI2 already has F10.7.

    Returns:
        pd.DataFrame with DatetimeIndex (UTC, hourly) and columns matching
        config.SW_FEATURES.
    """
    print("\n[Step 2] Cleaning NOAA space weather data...")

    omni = load_omni_hourly()
    celestrak = load_celestrak_daily()

    # OMNI2 already has all 6 SW features at hourly resolution
    # Verify completeness
    sw = omni[config.SW_FEATURES].copy()
    remaining_nans = sw.isna().sum()
    if remaining_nans.any():
        print(f"  WARNING: Remaining NaNs after cleaning: {remaining_nans[remaining_nans > 0].to_dict()}")
        # Final safety net: fill any stragglers
        sw = sw.ffill().bfill()

    print(f"  Output: {len(sw)} hourly rows, {sw.columns.tolist()}")
    print(f"  Period: {sw.index.min()} to {sw.index.max()}")
    print(f"  NaN check: {sw.isna().sum().sum()} remaining")

    return sw


if __name__ == "__main__":
    sw = build_sw_hourly()
    print(sw.describe())
