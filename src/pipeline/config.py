"""
SW-ADP Pipeline Configuration
Central constants, paths, and parameters for the data pipeline.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Raw file names (download scripts organize by source)
SEGMENTS_CSV = RAW_DATA_DIR / "opssat" / "segments.csv"
DATASET_CSV = RAW_DATA_DIR / "opssat" / "dataset.csv"
OMNI_HOURLY_CSV = RAW_DATA_DIR / "space_weather" / "omni_hourly.csv"
KP_F107_DAILY_CSV = RAW_DATA_DIR / "space_weather" / "kp_f107_daily.csv"
CELESTRAK_CSV = RAW_DATA_DIR / "space_weather" / "raw" / "celestrak_sw_5yr.csv"

# ── OPSSAT-AD Constants ────────────────────────────────────────────────
# Telemetry period (D-11)
OPSSAT_START = "2022-01-01"
OPSSAT_END = "2022-06-30"

# Channels to drop (D-10): too sparse for analysis
DROP_CHANNELS = ["CADC0886", "CADC0890"]

# Channel metadata
MAGNETOMETER_CHANNELS = ["CADC0872", "CADC0873", "CADC0874"]
PHOTODIODE_CHANNELS = ["CADC0884", "CADC0888", "CADC0892", "CADC0894"]
ALL_CHANNELS = MAGNETOMETER_CHANNELS + PHOTODIODE_CHANNELS  # 7 channels after D-10

# ── Space Weather Features ─────────────────────────────────────────────
SW_FEATURES = ["kp", "dst", "f10_7", "sw_speed", "sw_proton_density", "imf_bz_gsm"]

# Alignment strategy (D-06):
#   Forward-fill: discrete indices (Kp, F10.7, Dst) -- these are fixed-window values
#   Linear interpolation: continuous measurements (solar wind speed, proton density, IMF Bz)
SW_FFILL_COLS = ["kp", "dst", "f10_7"]
SW_INTERP_COLS = ["sw_speed", "sw_proton_density", "imf_bz_gsm"]

# ── Track A: Segment-Level Features ───────────────────────────────────
# Summary statistics to compute per segment from SW data
SW_SUMMARY_STATS = {
    "kp":                ["mean", "max", "std"],
    "dst":               ["mean", "min", "std"],
    "f10_7":             ["mean", "std"],
    "sw_speed":          ["mean", "max", "std"],
    "sw_proton_density": ["mean", "max", "std"],
    "imf_bz_gsm":       ["mean", "min", "std"],
}

# ── Track B: Time-Series Waveforms ────────────────────────────────────
SEQ_LEN = 128           # Fixed sequence length (configurable)
PAD_VALUE = 0.0         # Padding value for short segments
N_SW_FEATURES = len(SW_FEATURES)  # 6

# ── Reproducibility ───────────────────────────────────────────────────
RANDOM_SEED = 42
