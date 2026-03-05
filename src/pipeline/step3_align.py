"""
Step 3: Align OPSSAT telemetry segments with NOAA space weather data.

For each segment, produces:
  (a) SW summary statistics (Track A: segment-level features)
  (b) Per-timestep SW values (Track B: time-series input)

Alignment strategy (D-06):
  - SW data is hourly. OPSSAT telemetry is 1Hz or 5-second.
  - For each telemetry timestamp, find the matching hour in SW data.
  - Forward-fill is implicit: the hourly SW value applies to all
    telemetry timestamps within that hour.
"""
import pandas as pd
import numpy as np
from . import config


def load_segments() -> pd.DataFrame:
    """Load segments.csv, parse timestamps, drop sparse channels (D-10)."""
    df = pd.read_csv(config.SEGMENTS_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    n_before = len(df)
    seg_before = df["segment"].nunique()

    # Drop sparse channels (D-10)
    df = df[~df["channel"].isin(config.DROP_CHANNELS)]

    n_after = len(df)
    seg_after = df["segment"].nunique()

    print(f"  Segments: {seg_before} -> {seg_after} segments "
          f"({seg_before - seg_after} dropped from {config.DROP_CHANNELS})")
    print(f"  Rows: {n_before} -> {n_after}")

    return df


def load_dataset_features() -> pd.DataFrame:
    """Load dataset.csv (18 handcrafted features per segment), drop sparse channels."""
    df = pd.read_csv(config.DATASET_CSV)
    df = df[~df["channel"].isin(config.DROP_CHANNELS)]
    return df


def align_segment_sw(
    seg_df: pd.DataFrame,
    sw_hourly: pd.DataFrame,
) -> dict:
    """
    For a single segment's rows, align with hourly SW data.

    Returns dict with:
      - "sw_summary": dict of summary stats for Track A
      - "sw_timeseries": np.ndarray of shape [seg_len, N_sw] for Track B
      - "aligned_count": number of timestamps successfully aligned
      - "total_count": total timestamps in segment
    """
    timestamps = seg_df["timestamp"].sort_values()

    # Floor each telemetry timestamp to its hour to match SW index
    hours = timestamps.dt.floor("h")

    # Look up SW values for each hour
    # Use reindex to align, which handles missing hours gracefully
    unique_hours = hours.unique()
    sw_at_hours = sw_hourly.reindex(unique_hours)

    # Map back to per-timestamp level
    hour_to_idx = {h: i for i, h in enumerate(unique_hours)}
    hour_indices = hours.map(hour_to_idx).values
    sw_values = sw_at_hours.values[hour_indices]  # [seg_len, N_sw]

    # Count alignment success
    aligned_mask = ~np.isnan(sw_values).any(axis=1)
    aligned_count = int(aligned_mask.sum())

    # Compute summary stats for Track A
    sw_summary = {}
    for i, feat in enumerate(config.SW_FEATURES):
        col_vals = sw_values[:, i]
        valid = col_vals[~np.isnan(col_vals)]
        if len(valid) > 0:
            stat_funcs = config.SW_SUMMARY_STATS.get(feat, ["mean"])
            for stat in stat_funcs:
                if stat == "mean":
                    sw_summary[f"sw_{feat}_mean"] = np.mean(valid)
                elif stat == "max":
                    sw_summary[f"sw_{feat}_max"] = np.max(valid)
                elif stat == "min":
                    sw_summary[f"sw_{feat}_min"] = np.min(valid)
                elif stat == "std":
                    sw_summary[f"sw_{feat}_std"] = np.std(valid) if len(valid) > 1 else 0.0
        else:
            # No SW data available for this segment
            stat_funcs = config.SW_SUMMARY_STATS.get(feat, ["mean"])
            for stat in stat_funcs:
                sw_summary[f"sw_{feat}_{stat}"] = np.nan

    return {
        "sw_summary": sw_summary,
        "sw_timeseries": sw_values,  # [seg_len, N_sw]
        "aligned_count": aligned_count,
        "total_count": len(timestamps),
    }


def align_all_segments(
    segments_df: pd.DataFrame,
    sw_hourly: pd.DataFrame,
) -> tuple:
    """
    Align all segments with space weather data.

    Returns:
      - sw_summaries: dict[segment_id -> sw_summary_dict] for Track A
      - sw_timeseries: dict[segment_id -> np.ndarray] for Track B
      - alignment_report: DataFrame with alignment diagnostics
    """
    print("\n[Step 3] Aligning OPSSAT segments with space weather...")

    sw_summaries = {}
    sw_timeseries_dict = {}
    alignment_rows = []

    segment_ids = sorted(segments_df["segment"].unique())

    for seg_id in segment_ids:
        seg_rows = segments_df[segments_df["segment"] == seg_id]
        result = align_segment_sw(seg_rows, sw_hourly)

        sw_summaries[seg_id] = result["sw_summary"]
        sw_timeseries_dict[seg_id] = result["sw_timeseries"]

        alignment_rows.append({
            "segment": seg_id,
            "channel": seg_rows["channel"].iloc[0],
            "anomaly": seg_rows["anomaly"].iloc[0],
            "total_timestamps": result["total_count"],
            "aligned_timestamps": result["aligned_count"],
            "alignment_rate": result["aligned_count"] / result["total_count"]
                if result["total_count"] > 0 else 0,
        })

    report = pd.DataFrame(alignment_rows)

    perfect = (report["alignment_rate"] == 1.0).sum()
    partial = ((report["alignment_rate"] > 0) & (report["alignment_rate"] < 1.0)).sum()
    failed = (report["alignment_rate"] == 0).sum()

    print(f"  Aligned {len(segment_ids)} segments:")
    print(f"    Perfect (100%): {perfect}")
    print(f"    Partial:        {partial}")
    print(f"    Failed (0%):    {failed}")
    print(f"  Mean alignment rate: {report['alignment_rate'].mean():.4f}")

    return sw_summaries, sw_timeseries_dict, report
