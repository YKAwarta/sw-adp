#!/usr/bin/env python3
"""
SW-ADP Data Pipeline Runner
============================
Executes the full data pipeline from raw data to processed outputs.

Usage:
    python src/run_pipeline.py [--seq-len 128]

Stages:
    Step 2: Clean NOAA space weather data
    Step 3: Align OPSSAT segments with space weather
    Step 4A: Build Track A (segment-level features + SW stats)
    Step 4B: Build Track B (padded waveform tensors)
    Step 5: EDA and publication-quality figures

Inputs (data/):
    segments.csv, dataset.csv, omni_hourly.csv, kp_f107_daily.csv

Outputs:
    data/processed/track_a_telem_only.csv     -- Track A, Condition A
    data/processed/track_a_telem_sw.csv       -- Track A, Condition B
    data/processed/track_b_tensors.npz        -- Track B tensors (both conditions)
    data/processed/track_b_meta.csv           -- Track B segment metadata
    data/processed/normalization_stats.npz    -- Normalization parameters
    data/processed/alignment_report.csv       -- Alignment diagnostics
    figures/*.png                             -- EDA figures
"""
import sys
import time
import argparse
from pathlib import Path

# Ensure src/ is on the path so `pipeline` resolves as a package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import config
from pipeline.step2_clean_noaa import build_sw_hourly
from pipeline.step3_align import load_segments, align_all_segments
from pipeline.step4a_track_a import build_track_a, save_track_a
from pipeline.step4b_track_b import build_track_b, save_track_b
from pipeline.step5_eda import run_eda


def main(seq_len: int = config.SEQ_LEN):
    t0 = time.time()
    print("=" * 70)
    print("SW-ADP DATA PIPELINE")
    print("=" * 70)
    print(f"  Raw data:     {config.RAW_DATA_DIR}")
    print(f"  Processed:    {config.PROCESSED_DIR}")
    print(f"  Figures:      {config.FIGURES_DIR}")
    print(f"  Seq length:   {seq_len}")
    print(f"  Channels:     {config.ALL_CHANNELS}")
    print(f"  Dropped (D-10): {config.DROP_CHANNELS}")

    # Ensure output directories exist
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 2: Clean NOAA data ────────────────────────────────────────
    sw_hourly = build_sw_hourly()

    # ── Step 3: Align segments with SW ─────────────────────────────────
    segments_df = load_segments()
    sw_summaries, sw_timeseries, alignment_report = align_all_segments(
        segments_df, sw_hourly
    )

    # Save alignment report
    report_path = config.PROCESSED_DIR / "alignment_report.csv"
    alignment_report.to_csv(report_path, index=False)
    print(f"  Saved: {report_path}")

    # ── Step 4A: Track A ───────────────────────────────────────────────
    df_telem_only, df_telem_sw = build_track_a(sw_summaries)
    save_track_a(df_telem_only, df_telem_sw)

    # ── Step 4B: Track B ───────────────────────────────────────────────
    track_b_data = build_track_b(segments_df, sw_timeseries, seq_len=seq_len)
    save_track_b(track_b_data)

    # ── Step 5: EDA ────────────────────────────────────────────────────
    run_eda(segments_df, sw_hourly, track_b_data["meta"])

    # ── Done ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print("=" * 70)
    print(f"\nOutputs:")
    for f in sorted(config.PROCESSED_DIR.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:40s}  {size_kb:8.1f} KB")
    print(f"\nFigures:")
    for f in sorted(config.FIGURES_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SW-ADP Data Pipeline")
    parser.add_argument("--seq-len", type=int, default=config.SEQ_LEN,
                        help=f"Fixed sequence length for Track B (default: {config.SEQ_LEN})")
    args = parser.parse_args()
    main(seq_len=args.seq_len)
