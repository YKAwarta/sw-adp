"""
Step 5: Exploratory Data Analysis for feature justification.

Produces:
  1. Channel-SW correlation heatmap (publication quality)
  2. Anomaly vs. normal correlation comparison
  3. Summary statistics: anomaly rate per channel, segment lengths, temporal coverage
  4. Segment length distribution histogram
  5. Space weather conditions during OPSSAT period

Addresses Dr. Koubaa's feedback #2: feature justification.
Tests hypothesis D-07: magnetometers should show stronger SW correlation than photodiodes.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from . import config


# ── Publication style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
})

# Short display names for channels
CHANNEL_LABELS = {
    "CADC0872": "Mag-X\n(0872)",
    "CADC0873": "Mag-Y\n(0873)",
    "CADC0874": "Mag-Z\n(0874)",
    "CADC0884": "Photo-1\n(0884)",
    "CADC0888": "Photo-2\n(0888)",
    "CADC0892": "Photo-3\n(0892)",
    "CADC0894": "Photo-4\n(0894)",
}

SW_LABELS = {
    "kp": "Kp Index",
    "dst": "Dst Index",
    "f10_7": "F10.7 Flux",
    "sw_speed": "SW Speed",
    "sw_proton_density": "Proton\nDensity",
    "imf_bz_gsm": "IMF Bz",
}


def compute_channel_sw_correlations(
    segments_df: pd.DataFrame,
    sw_hourly: pd.DataFrame,
    method: str = "spearman",
) -> tuple:
    """
    Compute correlation between each channel's telemetry values and each SW index.

    For each segment, computes the mean telemetry value and the mean SW value
    during the segment's time window, then correlates across segments per channel.

    Returns:
        (corr_all, corr_anomaly, corr_normal): DataFrames of shape [N_channels, N_sw]
    """
    # Build segment-level summary: mean telemetry value + mean SW per segment
    rows = []
    for seg_id, grp in segments_df.groupby("segment"):
        grp = grp.sort_values("timestamp")
        ts = grp["timestamp"]
        hours = ts.dt.floor("h").unique()

        sw_match = sw_hourly.reindex(hours)
        sw_means = sw_match.mean()

        rows.append({
            "segment": seg_id,
            "channel": grp["channel"].iloc[0],
            "anomaly": grp["anomaly"].iloc[0],
            "telem_mean": grp["value"].mean(),
            "telem_std": grp["value"].std(),
            "telem_range": grp["value"].max() - grp["value"].min(),
            **{f"sw_{feat}": sw_means[feat] for feat in config.SW_FEATURES},
        })

    summary = pd.DataFrame(rows)

    def _compute_corr(df_sub):
        corr_matrix = pd.DataFrame(
            index=config.ALL_CHANNELS,
            columns=config.SW_FEATURES,
            dtype=float,
        )
        for ch in config.ALL_CHANNELS:
            ch_data = df_sub[df_sub["channel"] == ch]
            if len(ch_data) < 5:
                corr_matrix.loc[ch] = np.nan
                continue
            for feat in config.SW_FEATURES:
                x = ch_data["telem_mean"].values
                y = ch_data[f"sw_{feat}"].values
                valid = ~(np.isnan(x) | np.isnan(y))
                if valid.sum() < 5:
                    corr_matrix.loc[ch, feat] = np.nan
                    continue
                if method == "spearman":
                    r, _ = scipy_stats.spearmanr(x[valid], y[valid])
                else:
                    r, _ = scipy_stats.pearsonr(x[valid], y[valid])
                corr_matrix.loc[ch, feat] = r
        return corr_matrix

    corr_all = _compute_corr(summary)
    corr_anomaly = _compute_corr(summary[summary["anomaly"] == 1])
    corr_normal = _compute_corr(summary[summary["anomaly"] == 0])

    return corr_all, corr_anomaly, corr_normal, summary


def plot_correlation_heatmap(
    corr_all: pd.DataFrame,
    corr_anomaly: pd.DataFrame,
    corr_normal: pd.DataFrame,
):
    """Produce publication-quality correlation heatmaps."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

    datasets = [
        (corr_all, "All Segments"),
        (corr_normal, "Normal Segments"),
        (corr_anomaly, "Anomaly Segments"),
    ]

    for ax, (corr_df, title) in zip(axes, datasets):
        # Convert to float for plotting
        data = corr_df.astype(float).values

        ylabels = [CHANNEL_LABELS.get(ch, ch) for ch in corr_df.index]
        xlabels = [SW_LABELS.get(f, f) for f in corr_df.columns]

        im = ax.imshow(data, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")

        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Annotate cells with correlation values
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    txt = "-"
                else:
                    txt = f"{val:.2f}"
                color = "white" if abs(val) > 0.3 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, color=color)

        # Draw line separating magnetometers from photodiodes
        ax.axhline(y=2.5, color="black", linewidth=1.5, linestyle="--")

    # Add magnetometer/photodiode labels on leftmost axis
    axes[0].text(-0.6, 1.0, "Magnetometers", transform=axes[0].transData,
                 fontsize=9, rotation=90, va="center", ha="center",
                 fontweight="bold", style="italic")
    axes[0].text(-0.6, 4.5, "Photodiodes", transform=axes[0].transData,
                 fontsize=9, rotation=90, va="center", ha="center",
                 fontweight="bold", style="italic")

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Spearman Correlation (r)", fontsize=11)

    fig.suptitle(
        "Telemetry Channel -- Space Weather Index Correlations\n"
        "(OPSSAT-AD, Jan--Jun 2022)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    path = config.FIGURES_DIR / "channel_sw_correlation_heatmap.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_segment_length_distribution(meta: pd.DataFrame):
    """Histogram of segment lengths by anomaly status."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bins = np.arange(0, 550, 25)
    ax.hist(meta[meta["anomaly"] == 0]["length"], bins=bins, alpha=0.7,
            label="Normal", color="#2196F3", edgecolor="white")
    ax.hist(meta[meta["anomaly"] == 1]["length"], bins=bins, alpha=0.7,
            label="Anomaly", color="#F44336", edgecolor="white")

    ax.axvline(x=config.SEQ_LEN, color="black", linestyle="--", linewidth=1.5,
               label=f"Seq. length = {config.SEQ_LEN}")

    ax.set_xlabel("Segment Length (timesteps)")
    ax.set_ylabel("Count")
    ax.set_title("Segment Length Distribution by Anomaly Status")
    ax.legend()
    ax.set_xlim(0, 550)

    path = config.FIGURES_DIR / "segment_length_distribution.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_sw_timeseries(sw_hourly: pd.DataFrame):
    """Plot space weather conditions during OPSSAT period."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    idx = sw_hourly.index

    # Kp index
    axes[0].plot(idx, sw_hourly["kp"], color="#E65100", linewidth=0.5, alpha=0.8)
    axes[0].axhline(y=5, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].set_ylabel("Kp Index")
    axes[0].set_title("Space Weather Conditions During OPSSAT Telemetry Period (Jan--Jun 2022)",
                       fontsize=13, fontweight="bold")
    axes[0].text(idx[-1], 5.2, "Storm threshold (Kp=5)", fontsize=8,
                 color="red", ha="right", va="bottom")

    # Dst index
    axes[1].plot(idx, sw_hourly["dst"], color="#1565C0", linewidth=0.5, alpha=0.8)
    axes[1].axhline(y=-50, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[1].set_ylabel("Dst (nT)")
    axes[1].text(idx[-1], -52, "Moderate storm (Dst=-50)", fontsize=8,
                 color="red", ha="right", va="top")

    # Solar wind speed
    axes[2].plot(idx, sw_hourly["sw_speed"], color="#2E7D32", linewidth=0.5, alpha=0.8)
    axes[2].set_ylabel("SW Speed (km/s)")

    # F10.7
    axes[3].plot(idx, sw_hourly["f10_7"], color="#6A1B9A", linewidth=0.5, alpha=0.8)
    axes[3].set_ylabel("F10.7 (sfu)")
    axes[3].set_xlabel("Date (UTC)")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    path = config.FIGURES_DIR / "sw_conditions_opssat_period.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_anomaly_rate_per_channel(meta: pd.DataFrame):
    """Bar chart of anomaly rate per channel."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rates = meta.groupby("channel").agg(
        total=("anomaly", "count"),
        anomalies=("anomaly", "sum"),
    )
    rates["rate"] = rates["anomalies"] / rates["total"]
    rates = rates.loc[config.ALL_CHANNELS]  # Maintain order

    fig, ax = plt.subplots(figsize=(8, 4.5))

    xlabels = [CHANNEL_LABELS.get(ch, ch).replace("\n", " ") for ch in rates.index]
    colors = ["#E65100" if ch in config.MAGNETOMETER_CHANNELS else "#1565C0"
              for ch in rates.index]

    bars = ax.bar(range(len(rates)), rates["rate"], color=colors, edgecolor="white",
                  alpha=0.85)

    # Add count labels on bars
    for i, (_, row) in enumerate(rates.iterrows()):
        ax.text(i, row["rate"] + 0.01,
                f"{int(row['anomalies'])}/{int(row['total'])}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Anomaly Rate")
    ax.set_title("Anomaly Rate per Telemetry Channel")

    # Legend for channel types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E65100", alpha=0.85, label="Magnetometer"),
        Patch(facecolor="#1565C0", alpha=0.85, label="Photodiode"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.grid(axis="y", alpha=0.3)

    path = config.FIGURES_DIR / "anomaly_rate_per_channel.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def print_summary_statistics(meta: pd.DataFrame, sw_hourly: pd.DataFrame):
    """Print key statistics for the living document."""
    print("\n  --- Summary Statistics ---")
    print(f"  Total segments (after D-10 drops): {len(meta)}")
    print(f"  Anomaly segments: {meta['anomaly'].sum()} "
          f"({meta['anomaly'].mean()*100:.1f}%)")
    print(f"  Train segments: {(meta['train']==1).sum()}, "
          f"Test segments: {(meta['train']==0).sum()}")

    print(f"\n  Segment lengths:")
    print(f"    Min: {meta['length'].min()}, Max: {meta['length'].max()}")
    print(f"    Mean: {meta['length'].mean():.1f}, Median: {meta['length'].median():.1f}")
    print(f"    25th: {meta['length'].quantile(0.25):.0f}, "
          f"75th: {meta['length'].quantile(0.75):.0f}")
    pct_below_seq = (meta['length'] < config.SEQ_LEN).mean() * 100
    print(f"    Segments shorter than seq_len={config.SEQ_LEN}: {pct_below_seq:.1f}%")

    print(f"\n  Per-channel breakdown:")
    for ch in config.ALL_CHANNELS:
        ch_data = meta[meta["channel"] == ch]
        ch_type = "MAG" if ch in config.MAGNETOMETER_CHANNELS else "PHO"
        print(f"    {ch} ({ch_type}): {len(ch_data)} segs, "
              f"{int(ch_data['anomaly'].sum())} anom ({ch_data['anomaly'].mean()*100:.1f}%)")

    # SW during OPSSAT period
    print(f"\n  Space weather (OPSSAT period):")
    print(f"    Kp: mean={sw_hourly['kp'].mean():.2f}, max={sw_hourly['kp'].max():.1f}")
    print(f"    Dst: mean={sw_hourly['dst'].mean():.1f}, min={sw_hourly['dst'].min():.1f}")
    print(f"    F10.7: mean={sw_hourly['f10_7'].mean():.1f}")
    print(f"    Storm hours (Kp>=5): {(sw_hourly['kp']>=5).sum()}")
    print(f"    Storm hours (Dst<-50): {(sw_hourly['dst']<-50).sum()}")


def run_eda(
    segments_df: pd.DataFrame,
    sw_hourly: pd.DataFrame,
    meta: pd.DataFrame,
):
    """Run all EDA analyses and produce figures."""
    print("\n[Step 5] Running EDA for feature justification...")

    # 1. Correlation analysis
    print("\n  Computing channel-SW correlations...")
    corr_all, corr_anomaly, corr_normal, _ = compute_channel_sw_correlations(
        segments_df, sw_hourly, method="spearman"
    )

    print("\n  Correlation matrix (all segments, Spearman):")
    print(corr_all.to_string())

    # D-07 test: magnetometers vs photodiodes
    mag_corr = corr_all.loc[config.MAGNETOMETER_CHANNELS].abs().mean().mean()
    pho_corr = corr_all.loc[config.PHOTODIODE_CHANNELS].abs().mean().mean()
    print(f"\n  D-07 Hypothesis Test:")
    print(f"    Mean |correlation| magnetometers: {mag_corr:.4f}")
    print(f"    Mean |correlation| photodiodes:   {pho_corr:.4f}")
    if mag_corr > pho_corr:
        print(f"    Result: SUPPORTED -- magnetometers show stronger SW correlation")
    else:
        print(f"    Result: NOT SUPPORTED -- photodiodes show equal or stronger correlation")

    # 2. Plots
    plot_correlation_heatmap(corr_all, corr_anomaly, corr_normal)
    plot_segment_length_distribution(meta)
    plot_sw_timeseries(sw_hourly)
    plot_anomaly_rate_per_channel(meta)

    # 3. Summary stats
    print_summary_statistics(meta, sw_hourly)

    return corr_all, corr_anomaly, corr_normal
