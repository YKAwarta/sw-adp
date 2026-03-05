"""
Step 4A: Build Track A -- segment-level feature matrices for IF/MP baselines.

Condition A (telemetry-only): 18 handcrafted OPSSAT-AD features
Condition B (telemetry + SW):  18 features + SW summary statistics

Both conditions include: segment ID, channel, anomaly label, train/test split.
"""
import pandas as pd
import numpy as np
from . import config
from .step3_align import load_dataset_features


def build_track_a(
    sw_summaries: dict,
) -> tuple:
    """
    Build Track A feature DataFrames for both experimental conditions.

    Returns:
        (df_telem_only, df_telem_sw): Two DataFrames with identical index/metadata
        but different feature sets.
    """
    print("\n[Step 4A] Building Track A: segment-level feature matrices...")

    # Load the 18 handcrafted features from OPSSAT-AD benchmark
    feat_df = load_dataset_features()

    # Metadata columns (not features)
    meta_cols = ["segment", "anomaly", "train", "channel", "sampling"]
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]
    print(f"  OPSSAT-AD features ({len(feature_cols)}): {feature_cols}")

    # Build SW summary DataFrame from alignment results
    sw_rows = []
    for seg_id, summary in sw_summaries.items():
        row = {"segment": seg_id}
        row.update(summary)
        sw_rows.append(row)
    sw_df = pd.DataFrame(sw_rows)

    # Merge features with SW summaries
    merged = feat_df.merge(sw_df, on="segment", how="left")

    sw_feature_cols = [c for c in sw_df.columns if c != "segment"]
    print(f"  SW summary features ({len(sw_feature_cols)}): {sw_feature_cols}")

    # Check for any segments that couldn't be aligned
    n_missing_sw = merged[sw_feature_cols].isna().any(axis=1).sum()
    if n_missing_sw > 0:
        print(f"  WARNING: {n_missing_sw} segments have missing SW data. Filling with column means.")
        for col in sw_feature_cols:
            merged[col] = merged[col].fillna(merged[col].mean())

    # Condition A: telemetry-only features
    df_telem_only = merged[meta_cols + feature_cols].copy()

    # Condition B: telemetry + SW features
    df_telem_sw = merged[meta_cols + feature_cols + sw_feature_cols].copy()

    print(f"  Condition A shape: {df_telem_only.shape} "
          f"({len(feature_cols)} features)")
    print(f"  Condition B shape: {df_telem_sw.shape} "
          f"({len(feature_cols) + len(sw_feature_cols)} features)")

    # Quick sanity check
    n_train = (merged["train"] == 1).sum()
    n_test = (merged["train"] == 0).sum()
    n_anom = (merged["anomaly"] == 1).sum()
    n_norm = (merged["anomaly"] == 0).sum()
    print(f"  Train/Test split: {n_train}/{n_test}")
    print(f"  Anomaly/Normal: {n_anom}/{n_norm}")

    return df_telem_only, df_telem_sw


def save_track_a(df_telem_only: pd.DataFrame, df_telem_sw: pd.DataFrame):
    """Save Track A outputs to data/processed/."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    path_a = config.PROCESSED_DIR / "track_a_telem_only.csv"
    path_b = config.PROCESSED_DIR / "track_a_telem_sw.csv"

    df_telem_only.to_csv(path_a, index=False)
    df_telem_sw.to_csv(path_b, index=False)

    print(f"  Saved: {path_a}")
    print(f"  Saved: {path_b}")
