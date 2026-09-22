from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import module_00_style_library as sl  # noqa: E402

FEATURE_NAMES = ["total_power", "low_band_power", "high_band_power",
                  "spectral_centroid", "peak_to_mean", "log_log_slope"]
MODE_ORDER = ["healthy", "sensor_degradation", "radiation_burst", "coolant_anomaly",
              "foreign_material", "calibration_drift"]


def _spectral_features_from_scratch(S: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Standalone re-implementation of detector.spectral_features (kept
    self-contained here so this figure module has no hard runtime
    dependency on the simulation package, only on the CSV outputs)."""
    S = np.atleast_2d(S)
    logf = np.log10(freq)
    total = S.sum(axis=1)
    low_band = S[:, freq < 1.0].sum(axis=1)
    high_band = S[:, freq >= 1.0].sum(axis=1)
    centroid = (S * logf).sum(axis=1) / np.clip(S.sum(axis=1), 1e-30, None)
    peak_ratio = S.max(axis=1) / np.clip(S.mean(axis=1), 1e-30, None)
    logS = np.log10(np.clip(S, 1e-30, None))
    X = np.vstack([logf, np.ones_like(logf)]).T
    slopes = np.array([np.linalg.lstsq(X, row, rcond=None)[0][0] for row in logS])
    return np.vstack([total, low_band, high_band, centroid, peak_ratio, slopes]).T


def _rebuild_feature_table(noise_df: pd.DataFrame, n_per_class: int = 250,
                            seed: int = 777) -> pd.DataFrame:
    """
    Re-derive a labelled classical-feature table directly from the raw
    per-stage noise-spectrum snapshots (01_squid_noise_spectra.csv), by
    pairing each of the 20 repeated snapshots per lifecycle stage with
    small multiplicative/additive perturbations representative of the five
    failure modes -- i.e. this reproduces the SAME feature-extraction
    function used by the anomaly detector (detector.spectral_features),
    applied here directly to the saved fresh/pre-failure spectra so this
    figure needs no re-import of the simulation package.
    """
    rng = np.random.default_rng(seed)
    freq = np.sort(noise_df.freq_hz.unique())
    fresh_pivot = noise_df[noise_df.stage == "fresh"].pivot_table(
        index="rep", columns="freq_hz", values="S_phi_Phi0^2_per_Hz")
    prefail_pivot = noise_df[noise_df.stage == "7yr_prefail"].pivot_table(
        index="rep", columns="freq_hz", values="S_phi_Phi0^2_per_Hz")
    fresh_specs = fresh_pivot.to_numpy()
    prefail_specs = prefail_pivot.to_numpy()

    def sample_base(n):
        idx = rng.integers(0, len(fresh_specs), size=n)
        base = fresh_specs[idx].copy()
        jitter = np.exp(rng.normal(0, 0.05, size=base.shape))
        return base * jitter

    rows_S, rows_label = [], []
    healthy = sample_base(n_per_class)
    rows_S.append(healthy); rows_label += ["healthy"] * n_per_class

    deg_idx = rng.integers(0, len(prefail_specs), size=n_per_class)
    deg = prefail_specs[deg_idx] * np.exp(rng.normal(0, 0.05, size=(n_per_class, len(freq))))
    rows_S.append(deg); rows_label += ["sensor_degradation"] * n_per_class

    burst = sample_base(n_per_class) * rng.uniform(8, 20, size=(n_per_class, 1))
    rows_S.append(burst); rows_label += ["radiation_burst"] * n_per_class

    coolant = sample_base(n_per_class) * rng.uniform(1.1, 1.6, size=(n_per_class, 1))
    rows_S.append(coolant); rows_label += ["coolant_anomaly"] * n_per_class

    foreign = sample_base(n_per_class)
    f0 = 10 ** rng.uniform(1.0, 2.5, size=n_per_class)
    for i in range(n_per_class):
        width = f0[i] * 0.15
        bump = rng.uniform(3, 10) * foreign[i].max() * np.exp(-0.5 * ((freq - f0[i]) / width) ** 2)
        foreign[i] = foreign[i] + bump
    rows_S.append(foreign); rows_label += ["foreign_material"] * n_per_class

    drift = sample_base(n_per_class) * rng.uniform(2.5, 6.0, size=(n_per_class, 1))
    rows_S.append(drift); rows_label += ["calibration_drift"] * n_per_class

    S_all = np.vstack(rows_S)
    feats = _spectral_features_from_scratch(S_all, freq)
    df = pd.DataFrame(feats, columns=FEATURE_NAMES)
    df["failure_mode"] = rows_label
    return df


def _panel_a_corr_dendro(fig, ax, feat_df: pd.DataFrame) -> None:
    corr = feat_df[FEATURE_NAMES].corr(method="pearson")
    dist = 1 - corr.abs()
    condensed = squareform(dist.to_numpy(), checks=False)
    Z = hierarchy.linkage(condensed, method="average")
    order = hierarchy.leaves_list(Z)
    ordered_names = [FEATURE_NAMES[i] for i in order]
    corr_ordered = corr.loc[ordered_names, ordered_names]
    im = ax.imshow(corr_ordered, cmap=sl.CMAP_DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(ordered_names))); ax.set_xticklabels(ordered_names, rotation=45, ha="right", fontsize=5.6)
    ax.set_yticks(range(len(ordered_names))); ax.set_yticklabels(ordered_names, fontsize=5.6)
    for i in range(len(ordered_names)):
        for j in range(len(ordered_names)):
            ax.text(j, i, f"{corr_ordered.iloc[i, j]:.2f}", ha="center", va="center", fontsize=4.8,
                    color="white" if abs(corr_ordered.iloc[i, j]) > 0.6 else "black")
    sl.styled_colorbar(im, ax, "Pearson $r$", fig)


def _panel_b_pca(ax, feat_df: pd.DataFrame) -> None:
    X = StandardScaler().fit_transform(feat_df[FEATURE_NAMES].to_numpy())
    pca = PCA(n_components=2, random_state=0).fit(X)
    Z = pca.transform(X)
    for mode in MODE_ORDER:
        mask = feat_df.failure_mode == mode
        color = sl.ROLE_COLOR["healthy"] if mode == "healthy" else sl.FAILURE_MODE_COLORS[mode]
        label = "Healthy" if mode == "healthy" else sl.FAILURE_MODE_LABELS[mode]
        ax.scatter(Z[mask, 0], Z[mask, 1], s=8, color=color, alpha=0.65, linewidth=0,
                   label=label, rasterized=True)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
    ax.legend(fontsize=4.8, loc="best", ncol=1, markerscale=1.6)
    return pca


def _panel_c_tsne(ax, feat_df: pd.DataFrame, seed: int = 0) -> None:
    X = StandardScaler().fit_transform(feat_df[FEATURE_NAMES].to_numpy())
    tsne = TSNE(n_components=2, perplexity=25, random_state=seed, init="pca", learning_rate="auto")
    Z = tsne.fit_transform(X)
    for mode in MODE_ORDER:
        mask = feat_df.failure_mode == mode
        color = sl.ROLE_COLOR["healthy"] if mode == "healthy" else sl.FAILURE_MODE_COLORS[mode]
        ax.scatter(Z[mask, 0], Z[mask, 1], s=8, color=color, alpha=0.65, linewidth=0, rasterized=True)
    ax.set_xlabel("t-SNE dim. 1"); ax.set_ylabel("t-SNE dim. 2")
    ax.set_xticks([]); ax.set_yticks([])


def _panel_d_scree(ax, feat_df: pd.DataFrame) -> None:
    X = StandardScaler().fit_transform(feat_df[FEATURE_NAMES].to_numpy())
    pca = PCA(n_components=len(FEATURE_NAMES), random_state=0).fit(X)
    var = pca.explained_variance_ratio_ * 100
    x = np.arange(1, len(var) + 1)
    ax.bar(x, var, color=sl.OKABE_ITO["sky_blue"], edgecolor="white", linewidth=0.5, label="Individual")
    ax2 = ax.twinx()
    ax2.plot(x, np.cumsum(var), color=sl.OKABE_ITO["vermillion"], marker="o", markersize=3.5, lw=1.4,
             label="Cumulative")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative variance (%)", color=sl.OKABE_ITO["vermillion"])
    ax2.tick_params(axis="y", colors=sl.OKABE_ITO["vermillion"])
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_xticks(x)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6.0, loc="center right")


def build_figure(bundle) -> str:
    sl.print_section_header("FIGURE 7: Spectral Feature-Space Analysis (Correlation / PCA / t-SNE)")
    feat_df = _rebuild_feature_table(bundle.noise_spectra)
    sl.print_numeric_summary(feat_df, columns=FEATURE_NAMES, label="Re-derived classical spectral features")
    print(f"  Feature table: {len(feat_df)} samples x {len(FEATURE_NAMES)} features, "
          f"{feat_df.failure_mode.nunique()} classes")

    fig = plt.figure(figsize=sl.figsize_grid(2, 2, panel_w=2.9, panel_h=2.35))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.42, hspace=0.55,
                            left=0.09, right=0.95, top=0.90, bottom=0.12)

    ax_a = fig.add_subplot(gs[0, 0]); _panel_a_corr_dendro(fig, ax_a, feat_df); sl.label_panel(ax_a, "a")
    ax_b = fig.add_subplot(gs[0, 1]); _panel_b_pca(ax_b, feat_df); sl.label_panel(ax_b, "b")
    ax_c = fig.add_subplot(gs[1, 0]); _panel_c_tsne(ax_c, feat_df); sl.label_panel(ax_c, "c")
    ax_d = fig.add_subplot(gs[1, 1]); _panel_d_scree(ax_d, feat_df); sl.label_panel(ax_d, "d")

    fig.suptitle("Figure 7. Classical spectral feature-space analysis: correlation, PCA and t-SNE",
                  x=0.09, ha="left", fontsize=10.2, fontweight="bold", y=0.995)

    path = sl.save_publication_figure(
        fig, "Figure_07_feature_correlation_analysis",
        "Spectral feature-space analysis (correlation / PCA / t-SNE)",
        n_panels=4, n_data_points=len(feat_df) * len(FEATURE_NAMES),
        notes="scipy.cluster.hierarchy dendrogram ordering; sklearn PCA and t-SNE embeddings "
              "of the classical spectral-feature space (detector.py's spectral_features).")
    plt.close(fig)
    return path


if __name__ == "__main__":
    import module_01_data_loader as dl
    bundle = dl.load_all_datasets()
    build_figure(bundle)
