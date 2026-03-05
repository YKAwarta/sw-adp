"""
Step 4B: Build Track B -- padded waveform tensors for LSTM-AE / TCN-AE.

Each segment is a univariate time series. We pad/truncate to a fixed
sequence length and optionally append per-timestep SW features.

Condition A (telemetry-only): tensor shape [N, seq_len, 1]
Condition B (telemetry + SW):  tensor shape [N, seq_len, 1 + N_sw]

Padding strategy:
  - Segments shorter than seq_len: zero-pad at the END (post-padding)
  - Segments longer than seq_len: truncate from the END (keep the start)
  - A separate mask array tracks which timesteps are real vs. padded
"""
import pandas as pd
import numpy as np
from . import config


def normalize_segments(
    segments_df: pd.DataFrame,
) -> dict:
    """
    Extract per-segment raw time series, sorted by timestamp.
    Also compute per-channel normalization stats from TRAINING data only.

    Returns:
        dict with:
          - "series": dict[seg_id -> np.ndarray of values]
          - "channel_stats": dict[channel -> (mean, std)] from train set
          - "meta": DataFrame with segment metadata
    """
    # Build metadata
    meta_rows = []
    series = {}

    for seg_id, grp in segments_df.groupby("segment"):
        grp = grp.sort_values("timestamp")
        series[seg_id] = grp["value"].values

        meta_rows.append({
            "segment": seg_id,
            "channel": grp["channel"].iloc[0],
            "anomaly": grp["anomaly"].iloc[0],
            "train": grp["train"].iloc[0],
            "sampling": grp["sampling"].iloc[0],
            "length": len(grp),
        })

    meta = pd.DataFrame(meta_rows)

    # Compute per-channel normalization from TRAINING set only
    train_segments = meta[meta["train"] == 1]["segment"].values
    channel_stats = {}
    for ch in config.ALL_CHANNELS:
        ch_segs = meta[(meta["channel"] == ch) & (meta["train"] == 1)]["segment"].values
        if len(ch_segs) == 0:
            channel_stats[ch] = (0.0, 1.0)
            continue
        all_vals = np.concatenate([series[s] for s in ch_segs])
        channel_stats[ch] = (float(np.mean(all_vals)), float(np.std(all_vals) + 1e-8))

    return {"series": series, "channel_stats": channel_stats, "meta": meta}


def build_track_b(
    segments_df: pd.DataFrame,
    sw_timeseries: dict,
    seq_len: int = config.SEQ_LEN,
) -> dict:
    """
    Build Track B tensors for both experimental conditions.

    Args:
        segments_df: Raw segments DataFrame (already filtered to valid channels)
        sw_timeseries: dict[seg_id -> np.ndarray shape [seg_len, N_sw]] from alignment
        seq_len: Fixed sequence length for padding/truncation

    Returns:
        dict with keys:
          - "X_telem":   np.ndarray [N, seq_len, 1]          (Condition A)
          - "X_telem_sw": np.ndarray [N, seq_len, 1+N_sw]    (Condition B)
          - "y":          np.ndarray [N]                       (binary labels)
          - "mask":       np.ndarray [N, seq_len]              (1=real, 0=padded)
          - "train_mask": np.ndarray [N]                       (1=train, 0=test)
          - "meta":       DataFrame with segment metadata
          - "channel_stats": dict for denormalization
          - "sw_stats":   dict with (mean, std) per SW feature for normalization
    """
    print(f"\n[Step 4B] Building Track B: padded waveform tensors (seq_len={seq_len})...")

    extracted = normalize_segments(segments_df)
    series = extracted["series"]
    channel_stats = extracted["channel_stats"]
    meta = extracted["meta"]

    segment_ids = meta["segment"].values
    N = len(segment_ids)

    # Compute SW normalization stats from training segments
    train_sw_arrays = []
    for seg_id in meta[meta["train"] == 1]["segment"].values:
        if seg_id in sw_timeseries:
            arr = sw_timeseries[seg_id]
            valid_mask = ~np.isnan(arr).any(axis=1)
            if valid_mask.any():
                train_sw_arrays.append(arr[valid_mask])

    if train_sw_arrays:
        all_sw = np.concatenate(train_sw_arrays, axis=0)
        sw_mean = np.mean(all_sw, axis=0)
        sw_std = np.std(all_sw, axis=0) + 1e-8
    else:
        sw_mean = np.zeros(config.N_SW_FEATURES)
        sw_std = np.ones(config.N_SW_FEATURES)

    sw_stats = {"mean": sw_mean, "std": sw_std}

    # Allocate output arrays
    X_telem = np.full((N, seq_len, 1), config.PAD_VALUE, dtype=np.float32)
    X_telem_sw = np.full((N, seq_len, 1 + config.N_SW_FEATURES),
                         config.PAD_VALUE, dtype=np.float32)
    mask = np.zeros((N, seq_len), dtype=np.float32)
    y = np.zeros(N, dtype=np.int64)
    train_mask = np.zeros(N, dtype=np.int64)

    truncated_count = 0
    padded_count = 0

    for i, seg_id in enumerate(segment_ids):
        vals = series[seg_id]
        ch = meta.iloc[i]["channel"]
        ch_mean, ch_std = channel_stats[ch]

        # Z-score normalize the telemetry values
        vals_norm = (vals - ch_mean) / ch_std

        seg_len_actual = len(vals_norm)

        # Pad or truncate
        if seg_len_actual >= seq_len:
            vals_used = vals_norm[:seq_len]
            mask[i, :] = 1.0
            if seg_len_actual > seq_len:
                truncated_count += 1
        else:
            vals_used = vals_norm
            mask[i, :seg_len_actual] = 1.0
            padded_count += 1

        L = len(vals_used)
        X_telem[i, :L, 0] = vals_used

        # SW timeseries for this segment
        if seg_id in sw_timeseries:
            sw_vals = sw_timeseries[seg_id]  # [seg_len_actual, N_sw]
            # Normalize SW values
            sw_norm = (sw_vals - sw_mean) / sw_std
            # Handle any NaN from missing alignment
            sw_norm = np.nan_to_num(sw_norm, nan=0.0)
            sw_used = sw_norm[:seq_len] if seg_len_actual >= seq_len else sw_norm
            L_sw = len(sw_used)
            X_telem_sw[i, :L_sw, 0] = vals_used[:L_sw]
            X_telem_sw[i, :L_sw, 1:] = sw_used[:L_sw]
        else:
            X_telem_sw[i, :L, 0] = vals_used

        y[i] = meta.iloc[i]["anomaly"]
        train_mask[i] = meta.iloc[i]["train"]

    print(f"  Total segments: {N}")
    print(f"  Padded (< {seq_len}): {padded_count}")
    print(f"  Truncated (> {seq_len}): {truncated_count}")
    print(f"  Exact fit (= {seq_len}): {N - padded_count - truncated_count}")
    print(f"  X_telem shape:    {X_telem.shape}")
    print(f"  X_telem_sw shape: {X_telem_sw.shape}")
    print(f"  Labels: {int(y.sum())} anomaly, {int((1-y).sum())} normal")
    print(f"  Train/Test: {int(train_mask.sum())}/{int((1-train_mask).sum())}")

    return {
        "X_telem": X_telem,
        "X_telem_sw": X_telem_sw,
        "y": y,
        "mask": mask,
        "train_mask": train_mask,
        "meta": meta,
        "channel_stats": channel_stats,
        "sw_stats": sw_stats,
    }


def save_track_b(data: dict):
    """Save Track B outputs as .npz archive to data/processed/."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    path = config.PROCESSED_DIR / "track_b_tensors.npz"

    np.savez_compressed(
        path,
        X_telem=data["X_telem"],
        X_telem_sw=data["X_telem_sw"],
        y=data["y"],
        mask=data["mask"],
        train_mask=data["train_mask"],
    )
    print(f"  Saved tensors: {path}")

    # Save metadata separately as CSV (can't put DataFrame in npz)
    meta_path = config.PROCESSED_DIR / "track_b_meta.csv"
    data["meta"].to_csv(meta_path, index=False)
    print(f"  Saved metadata: {meta_path}")

    # Save normalization stats
    stats_path = config.PROCESSED_DIR / "normalization_stats.npz"
    np.savez(
        stats_path,
        sw_mean=data["sw_stats"]["mean"],
        sw_std=data["sw_stats"]["std"],
        channel_names=np.array(list(data["channel_stats"].keys())),
        channel_means=np.array([v[0] for v in data["channel_stats"].values()]),
        channel_stds=np.array([v[1] for v in data["channel_stats"].values()]),
    )
    print(f"  Saved normalization stats: {stats_path}")
