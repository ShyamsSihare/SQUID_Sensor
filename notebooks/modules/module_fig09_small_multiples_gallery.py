from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import module_00_style_library as sl  # noqa: E402

MODE_ORDER = ["sensor_degradation", "radiation_burst", "coolant_anomaly",
              "foreign_material", "calibration_drift"]
STAGE_ORDER = ["fresh", "2yr", "5yr_precursor", "7yr_prefail"]
STAGE_SHORT = {"fresh": "Fresh", "2yr": "2 yr", "5yr_precursor": "5 yr", "7yr_prefail": "Pre-fail"}


def _build_gallery(fig, outer_gs, noise_df: pd.DataFrame, n_reps_per_stage: int = 5) -> None:
    inner = outer_gs.subgridspec(4, n_reps_per_stage, wspace=0.08, hspace=0.15)
    rng = np.random.default_rng(3)
    for si, stage in enumerate(STAGE_ORDER):
        stage_df = noise_df[noise_df.stage == stage]
        reps = sorted(stage_df.rep.unique())
        chosen = rng.choice(reps, size=n_reps_per_stage, replace=False)
        for ri, rep in enumerate(sorted(chosen)):
            ax = fig.add_subplot(inner[si, ri])
            g = stage_df[stage_df.rep == rep].sort_values("freq_hz")
            ax.plot(g.freq_hz, g.sqrt_S_phi_Phi0_per_sqrtHz * 1e6,
                    color=sl.STAGE_COLORS[stage] if hasattr(sl, "STAGE_COLORS") else
                    {"fresh": sl.ROLE_COLOR["fresh"], "2yr": sl.ROLE_COLOR["stage2"],
                     "5yr_precursor": sl.ROLE_COLOR["stage3"], "7yr_prefail": sl.ROLE_COLOR["stage4"]}[stage],
                    lw=0.8)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(1e-2, 1e4)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            if ri == 0:
                ax.set_ylabel(STAGE_SHORT[stage], fontsize=6.2, rotation=0, ha="right", va="center", labelpad=4)
            if si == 0:
                ax.set_title(f"snapshot {ri+1}", fontsize=5.8)


def _mannwhitney_tests(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Two-sided Mann-Whitney U tests: healthy vs. each failure mode, for
    BOTH detectors, with Bonferroni correction across the 10 resulting
    tests (5 modes x 2 detectors), plus rank-biserial correlation as a
    non-parametric effect size."""
    healthy_q = scores_df[scores_df.failure_mode == "healthy"].quantum_relentropy_score
    healthy_c = scores_df[scores_df.failure_mode == "healthy"].classical_ocsvm_score
    rows = []
    n_tests = len(MODE_ORDER) * 2
    for mode in MODE_ORDER:
        sub = scores_df[scores_df.failure_mode == mode]
        for det, healthy_ref, col in [("quantum", healthy_q, "quantum_relentropy_score"),
                                       ("classical", healthy_c, "classical_ocsvm_score")]:
            u_stat, p_val = stats.mannwhitneyu(sub[col], healthy_ref, alternative="two-sided")
            n1, n2 = len(sub[col]), len(healthy_ref)
            rank_biserial = abs(1 - (2 * u_stat) / (n1 * n2))
            p_corrected = min(1.0, p_val * n_tests)
            rows.append(dict(failure_mode=mode, detector=det, U=u_stat, p_raw=p_val,
                              p_bonferroni=p_corrected, effect_size_rank_biserial=rank_biserial,
                              n1=n1, n2=n2))
    return pd.DataFrame(rows)


def _sig_stars(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def _panel_effect_sizes(ax, test_df: pd.DataFrame) -> None:
    x = np.arange(len(MODE_ORDER))
    w = 0.36
    q = test_df[test_df.detector == "quantum"].set_index("failure_mode").loc[MODE_ORDER]
    c = test_df[test_df.detector == "classical"].set_index("failure_mode").loc[MODE_ORDER]
    ax.bar(x - w / 2, q.effect_size_rank_biserial, width=w, color=sl.ROLE_COLOR["quantum"],
           edgecolor="white", linewidth=0.4, label="Quantum")
    ax.bar(x + w / 2, c.effect_size_rank_biserial, width=w, color=sl.ROLE_COLOR["classical"],
           edgecolor="white", linewidth=0.4, label="Classical")
    for i, mode in enumerate(MODE_ORDER):
        for xoff, row in [(-w / 2, q.loc[mode]), (w / 2, c.loc[mode])]:
            stars = _sig_stars(row.p_bonferroni)
            ax.text(i + xoff, max(row.effect_size_rank_biserial, 0) + 0.03, stars,
                    ha="center", fontsize=6.0)
    ax.set_xticks(x)
    ax.set_xticklabels([sl.FAILURE_MODE_LABELS[m] for m in MODE_ORDER], fontsize=5.4, rotation=28, ha="right")
    ax.set_ylabel("Rank-biserial effect size\n(vs. healthy, Mann-Whitney $U$)")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=6.2, loc="lower right")


def _panel_pvalue_heatmap(fig, ax, test_df: pd.DataFrame) -> None:
    pivot = test_df.pivot_table(index="detector", columns="failure_mode", values="p_bonferroni")
    pivot = pivot[MODE_ORDER]
    logp = -np.log10(np.clip(pivot.to_numpy(), 1e-300, None))
    im = ax.imshow(logp, cmap=sl.CMAP_SEQUENTIAL, aspect="auto")
    ax.set_xticks(range(len(MODE_ORDER)))
    ax.set_xticklabels([sl.FAILURE_MODE_LABELS[m] for m in MODE_ORDER], fontsize=5.4, rotation=28, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([d.capitalize() for d in pivot.index], fontsize=6.5)
    for i in range(logp.shape[0]):
        for j in range(logp.shape[1]):
            ax.text(j, i, _sig_stars(pivot.to_numpy()[i, j]), ha="center", va="center",
                    fontsize=6.0, color="white" if logp[i, j] > logp.max() * 0.5 else "black")
    sl.styled_colorbar(im, ax, r"$-\log_{10}(p_{\rm Bonferroni})$", fig)


def _panel_effect_vs_auc(ax, test_df: pd.DataFrame, perf_by_mode: pd.DataFrame) -> None:
    q = test_df[test_df.detector == "quantum"].set_index("failure_mode").loc[MODE_ORDER]
    perf = perf_by_mode.set_index("failure_mode").loc[MODE_ORDER]
    for mode in MODE_ORDER:
        ax.scatter(q.loc[mode, "effect_size_rank_biserial"], perf.loc[mode, "AUC_quantum"],
                   color=sl.FAILURE_MODE_COLORS[mode], s=40, edgecolor="white", linewidth=0.5,
                   label=sl.FAILURE_MODE_LABELS[mode])
    r, p = stats.pearsonr(q.effect_size_rank_biserial, perf.AUC_quantum)
    ax.set_xlabel("Rank-biserial effect size (quantum)")
    ax.set_ylabel("AUC (quantum)")
    ax.legend(fontsize=5.2, loc="lower right")
    ax.text(0.05, 0.92, f"Pearson $r={r:.2f}$, $p={p:.3f}$", transform=ax.transAxes, fontsize=6.2)


def build_figure(bundle) -> str:
    sl.print_section_header("FIGURE 9: Small-Multiples Gallery & Rigorous Statistical Testing")
    noise_df = bundle.noise_spectra
    scores_df = bundle.anomaly_scores
    test_df = _mannwhitney_tests(scores_df)
    print("\n  Mann-Whitney U tests (healthy vs. each failure mode), Bonferroni-corrected:")
    print(test_df.to_string(index=False, float_format=lambda v: f"{v:.5g}"))

    fig = plt.figure(figsize=(sl.COLUMN_WIDTH_DOUBLE_IN, 7.6))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.15, 1.0], hspace=0.32,
                               top=0.93, bottom=0.07, left=0.09, right=0.97)

    _build_gallery(fig, outer[0], noise_df, n_reps_per_stage=5)
    fig.text(0.09, 0.955, "a", fontsize=plt.rcParams["font.size"] + 2, fontweight="bold")
    fig.text(0.50, 0.955, "Raw noise-spectrum snapshots (20 of 80 total), 4 lifecycle stages x 5 samples each",
              fontsize=7.5, ha="center")

    inner_bottom = outer[1].subgridspec(1, 3, wspace=0.55)
    ax_b = fig.add_subplot(inner_bottom[0, 0]); _panel_effect_sizes(ax_b, test_df); sl.label_panel(ax_b, "b")
    ax_c = fig.add_subplot(inner_bottom[0, 1]); _panel_pvalue_heatmap(fig, ax_c, test_df); sl.label_panel(ax_c, "c")
    ax_d = fig.add_subplot(inner_bottom[0, 2])
    _panel_effect_vs_auc(ax_d, test_df, bundle.perf_by_mode); sl.label_panel(ax_d, "d")

    fig.suptitle("Figure 9. Raw-spectrum small-multiples gallery and rigorous statistical hypothesis testing",
                  x=0.09, ha="left", fontsize=10.2, fontweight="bold", y=0.995)

    n_pts = 20 * len(noise_df[noise_df.stage == "fresh"].freq_hz.unique()) + len(scores_df)
    path = sl.save_publication_figure(
        fig, "Figure_09_small_multiples_gallery",
        "Small-multiples gallery and statistical hypothesis testing",
        n_panels=24, n_data_points=n_pts,
        notes="20-panel raw-spectrum gallery (audit view) + Mann-Whitney U tests, "
              "Bonferroni correction, rank-biserial effect sizes (scipy.stats).")
    plt.close(fig)
    return path


if __name__ == "__main__":
    import module_01_data_loader as dl
    bundle = dl.load_all_datasets()
    build_figure(bundle)
