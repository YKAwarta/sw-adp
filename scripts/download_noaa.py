"""
SW-ADP: Download Space Weather Indices
Sources:
  1. CelesTrak SW-Last5Years.csv -- Kp (3h), Ap, F10.7 (daily)
     https://celestrak.com/SpaceData/
  2. NASA OMNIWeb OMNI2 hourly   -- Dst (hourly), solar wind speed/density, IMF Bz
     https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/

Covers 2021-2026, which fully brackets the OPSSAT-AD telemetry period.

Usage:
    conda activate sw-adp
    python scripts/download_noaa.py
"""

import os
import sys
import urllib.request
from pathlib import Path
import io

DATA_DIR = Path("data") / "space_weather"

# ---------------------------------------------------------------------------
# Source 1: CelesTrak (Kp, Ap, F10.7)
# ---------------------------------------------------------------------------
CELESTRAK_URL = "https://celestrak.com/SpaceData/SW-Last5Years.csv"
CELESTRAK_FILE = "celestrak_sw_5yr.csv"

# ---------------------------------------------------------------------------
# Source 2: NASA OMNI2 hourly (Dst, solar wind, IMF Bz)
# Each yearly file: ~0.5 MB
# ---------------------------------------------------------------------------
OMNI_BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni"
OMNI_YEARS = [2022, 2023, 2024, 2025]  # bracket OPSSAT period with margin
OMNI_FILE = "omni2_hourly.csv"

# OMNI2 column definitions (1-indexed in docs, 0-indexed here)
# Full format: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text
OMNI_COLUMNS = {
    0: "year",
    1: "doy",          # Day of year (1-366)
    2: "hour",         # Hour (0-23)
    # IMF
    14: "imf_bz_gsm",  # IMF Bz GSM (nT) -- key driver of geomagnetic storms
    # Solar wind
    23: "sw_proton_density",  # Proton density (n/cc)
    24: "sw_speed",           # Solar wind speed (km/s)
    # Geomagnetic
    38: "kp",           # Kp * 10 (integer, divide by 10 for standard 0-9 scale)
    40: "dst",          # Dst (nT) -- hourly storm index
    # Solar
    50: "f10_7",        # F10.7 solar radio flux (sfu)
}

# OMNI fill values (missing data markers)
OMNI_FILL = {
    "imf_bz_gsm": 999.9,
    "sw_proton_density": 999.9,
    "sw_speed": 9999.0,
    "kp": 99,
    "dst": 99999,
    "f10_7": 999.9,
}


def download(url: str, dest: Path, label: str) -> bool:
    """Download a file with basic progress."""
    print(f"\n  Downloading: {label}")
    print(f"  URL: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size / 1024
        print(f"  [OK]   Saved: {dest.name} ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def parse_celestrak(filepath: Path, output_path: Path):
    """
    Parse CelesTrak SW CSV into a clean daily CSV.
    CelesTrak format has a multi-line header, then CSV data with columns:
    DATE, BSRN, ND, KP1..KP8, KP_SUM, AP1..AP8, AP_AVG, CP, C9, ISN,
    F10.7_OBS, F10.7_ADJ, ...
    """
    import pandas as pd

    print(f"\n  Parsing CelesTrak data...")

    # Read the file, skipping the metadata header lines
    lines = filepath.read_text().splitlines()

    # Find the CSV header line (starts with DATE)
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("DATE"):
            header_idx = i
            break

    if header_idx is None:
        print("  [FAIL] Could not find CSV header in CelesTrak file.")
        return False

    # Read from the header line onward
    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text))

    # Clean column names
    df.columns = df.columns.str.strip()

    # Parse date
    df["date"] = pd.to_datetime(df["DATE"])

    # Select and rename relevant columns
    # Kp values are stored as KP1..KP8 (8 three-hourly values per day)
    # These are in thirds: 0, 3, 7, 10, 13, ... representing 0.0, 0.3, 0.7, 1.0, ...
    kp_cols = [c for c in df.columns if c.startswith("KP") and c != "KP_SUM"]
    ap_cols = [c for c in df.columns if c.startswith("AP") and c != "AP_AVG"]

    # Build clean output
    out = pd.DataFrame()
    out["date"] = df["date"]

    # Daily Kp: use KP_SUM / 8 for daily average, and keep individual 3h values
    for col in kp_cols:
        # CelesTrak Kp values are integer*10 (e.g., 30 = Kp 3.0)
        out[col.lower()] = df[col] / 10.0

    out["kp_avg"] = df["KP_SUM"] / 80.0  # sum of 8 values, each *10
    out["ap_avg"] = df["AP_AVG"]

    # F10.7
    if "F10.7_OBS" in df.columns:
        out["f10_7"] = pd.to_numeric(df["F10.7_OBS"], errors="coerce")
    elif "F10.7_ADJ" in df.columns:
        out["f10_7"] = pd.to_numeric(df["F10.7_ADJ"], errors="coerce")

    # ISN (International Sunspot Number) -- bonus feature
    if "ISN" in df.columns:
        out["sunspot_number"] = pd.to_numeric(df["ISN"], errors="coerce")

    out.to_csv(output_path, index=False)
    print(f"  [OK]   Saved: {output_path.name}")
    print(f"         Rows: {len(out):,} | Date range: {out['date'].min()} to {out['date'].max()}")
    return True


def parse_omni2(data_dir: Path, output_path: Path):
    """
    Parse NASA OMNI2 yearly files into a clean hourly CSV.
    OMNI2 is fixed-width space-delimited, one row per hour.
    """
    import pandas as pd
    import numpy as np

    print(f"\n  Parsing OMNI2 hourly data...")

    all_rows = []
    for year in OMNI_YEARS:
        fpath = data_dir / f"omni2_{year}.dat"
        if not fpath.exists():
            print(f"  [WARN] Missing: {fpath.name}, skipping year {year}")
            continue

        # Read space-delimited fixed-width file
        with open(fpath) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 55:
                    continue

                row = {}
                for col_idx, col_name in OMNI_COLUMNS.items():
                    try:
                        val = float(parts[col_idx])
                        # Replace fill values with NaN
                        fill = OMNI_FILL.get(col_name)
                        if fill is not None and abs(val - fill) < 0.1:
                            val = float("nan")
                        row[col_name] = val
                    except (IndexError, ValueError):
                        row[col_name] = float("nan")

                all_rows.append(row)

    if not all_rows:
        print("  [FAIL] No OMNI2 data parsed.")
        return False

    df = pd.DataFrame(all_rows)

    # Build proper datetime
    df["year"] = df["year"].astype(int)
    df["doy"] = df["doy"].astype(int)
    df["hour"] = df["hour"].astype(int)

    df["datetime"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["doy"].astype(str),
        format="%Y-%j"
    ) + pd.to_timedelta(df["hour"], unit="h")

    # Scale Kp (stored as Kp*10 in OMNI2)
    df["kp"] = df["kp"] / 10.0

    # Select final columns
    out = df[["datetime", "kp", "dst", "f10_7", "sw_speed",
              "sw_proton_density", "imf_bz_gsm"]].copy()

    out.to_csv(output_path, index=False)
    print(f"  [OK]   Saved: {output_path.name}")
    print(f"         Rows: {len(out):,} | Date range: {out['datetime'].min()} to {out['datetime'].max()}")

    # Quick stats
    print(f"         Dst range: [{out['dst'].min():.0f}, {out['dst'].max():.0f}] nT")
    print(f"         Kp range:  [{out['kp'].min():.1f}, {out['kp'].max():.1f}]")
    print(f"         F10.7 range: [{out['f10_7'].min():.1f}, {out['f10_7'].max():.1f}] sfu")
    print(f"         Missing data: {out.isna().sum().to_dict()}")

    return True


def main():
    print("=" * 60)
    print("SW-ADP: SPACE WEATHER INDICES DOWNLOAD")
    print("=" * 60)
    print(f"Target directory: {DATA_DIR.resolve()}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)

    all_ok = True

    # --- Source 1: CelesTrak ---
    print("\n" + "-" * 60)
    print("SOURCE 1: CelesTrak (Kp, Ap, F10.7 -- daily)")
    print("-" * 60)

    celestrak_raw = raw_dir / CELESTRAK_FILE
    if not celestrak_raw.exists():
        if not download(CELESTRAK_URL, celestrak_raw, "CelesTrak SW-Last5Years"):
            all_ok = False
    else:
        print(f"  File exists: {celestrak_raw.name}, skipping download.")

    if celestrak_raw.exists():
        try:
            parse_celestrak(celestrak_raw, DATA_DIR / "kp_f107_daily.csv")
        except Exception as e:
            print(f"  [FAIL] Parse error: {e}")
            all_ok = False

    # --- Source 2: NASA OMNI2 ---
    print("\n" + "-" * 60)
    print("SOURCE 2: NASA OMNI2 (Dst, solar wind, IMF -- hourly)")
    print("-" * 60)

    for year in OMNI_YEARS:
        dest = raw_dir / f"omni2_{year}.dat"
        if not dest.exists():
            url = f"{OMNI_BASE_URL}/omni2_{year}.dat"
            if not download(url, dest, f"OMNI2 {year}"):
                # Non-fatal: future year files may not exist yet
                print(f"  [WARN] Could not download {year}. If this is a future year, this is expected.")
        else:
            print(f"  File exists: {dest.name}, skipping download.")

    try:
        parse_omni2(raw_dir, DATA_DIR / "omni_hourly.csv")
    except Exception as e:
        print(f"  [FAIL] Parse error: {e}")
        all_ok = False

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_ok:
        print("Space weather download complete.")
        print(f"Files in: {DATA_DIR.resolve()}")
        print()
        print("Output files:")
        print(f"  kp_f107_daily.csv  -- daily Kp (3h x8), Ap, F10.7 from CelesTrak")
        print(f"  omni_hourly.csv    -- hourly Dst, solar wind, IMF Bz from NASA OMNI2")
        print()
        print("Next step: python scripts/verify_data.py")
    else:
        print("Some downloads or parsing failed. Check errors above.")
        print("Note: 2025/2026 OMNI2 files may not be available yet;")
        print("      the script will still work with whatever years are available.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
