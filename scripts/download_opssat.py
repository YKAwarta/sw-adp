"""
SW-ADP: Download OPSSAT-AD Dataset from Zenodo
Source: https://zenodo.org/records/12588359
Paper:  Ruszczak et al., "The OPS-SAT benchmark for detecting anomalies
        in satellite telemetry", Scientific Data, 2025.

Files:
  - segments.csv  (18.0 MB) -- raw telemetry signals from ESA OPS-SAT
  - dataset.csv   (508 KB)  -- pre-extracted features per labeled segment

Usage:
    conda activate sw-adp
    python scripts/download_opssat.py
"""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data") / "opssat"
ZENODO_RECORD = "12588359"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

FILES = {
    "segments.csv": {
        "url": f"{BASE_URL}/segments.csv?download=1",
        "md5": "1f4b6ffd0359efac96a6698ff557e0f9",
        "size_mb": 18.0,
        "description": "Raw telemetry signals (9 channels, labeled segments)",
    },
    "dataset.csv": {
        "url": f"{BASE_URL}/dataset.csv?download=1",
        "md5": "c12fb83863afa4463dec07303a20bb14",
        "size_mb": 0.5,
        "description": "Pre-extracted features per segment (for benchmarking)",
    },
}


def md5_checksum(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, description: str, size_mb: float):
    """Download a file with progress reporting."""
    print(f"\n  Downloading: {dest.name}")
    print(f"  Description: {description}")
    print(f"  Expected size: ~{size_mb:.1f} MB")
    print(f"  Source: {url[:80]}...")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded / total_size * 100)
            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = "=" * filled + "-" * (bar_len - filled)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  [{bar}] {pct:5.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
        print()  # newline after progress bar
        return True
    except Exception as e:
        print(f"\n  [FAIL] Download error: {e}")
        return False


def verify_file(filepath: Path, expected_md5: str) -> bool:
    """Verify file integrity via MD5 checksum."""
    if not filepath.exists():
        print(f"  [FAIL] File not found: {filepath}")
        return False

    actual_md5 = md5_checksum(filepath)
    if actual_md5 == expected_md5:
        print(f"  [OK]   MD5 verified: {filepath.name}")
        return True
    else:
        print(f"  [FAIL] MD5 mismatch for {filepath.name}")
        print(f"         Expected: {expected_md5}")
        print(f"         Got:      {actual_md5}")
        return False


def main():
    print("=" * 60)
    print("SW-ADP: OPSSAT-AD DATASET DOWNLOAD")
    print("=" * 60)
    print(f"\nZenodo record: {ZENODO_RECORD}")
    print(f"Target directory: {DATA_DIR.resolve()}")

    # Create directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for filename, info in FILES.items():
        dest = DATA_DIR / filename

        # Skip if already downloaded and verified
        if dest.exists():
            print(f"\n  File exists: {filename}")
            if verify_file(dest, info["md5"]):
                print(f"  Skipping re-download.")
                continue
            else:
                print(f"  Re-downloading (checksum mismatch)...")

        # Download
        success = download_file(info["url"], dest, info["description"], info["size_mb"])
        if not success:
            all_ok = False
            continue

        # Verify
        if not verify_file(dest, info["md5"]):
            all_ok = False

    # Quick data inspection
    print("\n" + "-" * 60)
    print("QUICK DATA INSPECTION")
    print("-" * 60)

    try:
        import pandas as pd

        for filename in FILES:
            filepath = DATA_DIR / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                print(f"\n  {filename}:")
                print(f"    Rows: {len(df):,}")
                print(f"    Columns: {df.shape[1]}")
                print(f"    Column names: {list(df.columns)}")
                if "label" in df.columns:
                    label_counts = df["label"].value_counts()
                    print(f"    Label distribution:")
                    for label, count in label_counts.items():
                        print(f"      {label}: {count:,} ({count/len(df)*100:.1f}%)")
    except ImportError:
        print("  pandas not available; skipping data inspection.")
    except Exception as e:
        print(f"  Inspection error: {e}")

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("OPSSAT-AD download complete and verified.")
        print(f"Files in: {DATA_DIR.resolve()}")
        print("\nNext step: python scripts/download_noaa.py")
    else:
        print("Some downloads failed. Check errors above and retry.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
