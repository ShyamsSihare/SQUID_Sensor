from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
import module_00_style_library as sl  # noqa: E402

MODE_ORDER = ["sensor_degradation", "radiation_burst", "coolant_anomaly",
              "foreign_material", "calibration_drift"]

# Manuscript's own Table 6.2 (tab:anomaly_performance) values, reproduced here
# ONLY as a labelled reference overlay for visual comparison -- these are the
# manuscript's stated numbers, not our simulation output (see README.md
# Section 3 for the documented reasons those exact absolute numbers cannot be
# independently reproduced from the manuscript's own equations).
MANUSCRIPT_TABLE_TPR = {
    "sensor_degradation": 96.2, "radiation_burst": 99.8, "coolant_anomaly": 88.5,
    "foreign_material": 92.1, "calibration_drift": 94.4,
}


def _panel_a_roc(ax, roc_df: pd.DataFrame, perf_by_mode: pd.DataFrame) -> None:
    for det, color, key in [("quantum", sl.ROLE_COLOR["quantum"], "quantum"),
                             ("classical", sl.ROLE_COLOR["classical"], "classical")]:
        sub = roc_df[roc_df.detector == det].sort_values("fpr")
        auc = np.trapezoid(sub.tpr, sub.fpr)
        ax.plot(sub.fpr, sub.tpr, color=color, lw=1.5,
                label=f"{det.capitalize()} (AUC={auc:.3f})")
        ax.fill_between(sub.fpr, sub.tpr, sub.fpr, color=color, alpha=0.05)
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=6.3)
    axins = sl.inset_zoom(ax, bounds=[0.42, 0.08, 0.5, 0.5], xlim=(0, 0.05), ylim=(0.4, 1.0))
    for det, color in [("quantum", sl.ROLE_COLOR["quantum"]), ("classical", sl.ROLE_COLOR["classical"])]:
        sub = roc_df[roc_df.detector == det].sort_values("fpr")
        axins.plot(sub.fpr, sub.tpr, color=color, lw=1.1)


def _panel_b_tpr_bars(ax, perf: pd.DataFrame) -> None:
    x = np.arange(len(MODE_ORDER))
    w = 0.36
    ordered = perf.set_index("failure_mode").loc[MODE_ORDER].reset_index()
    ax.bar(x - w / 2, ordered.TPR_quantum_pct, width=w, color=sl.ROLE_COLOR["quantum"],
           label="Quantum (this simulation)", edgecolor="white", linewidth=0.4)
    ax.bar(x + w / 2, ordered.TPR_classical_pct, width=w, color=sl.ROLE_COLOR["classical"],
           label="Classical (this simulation)", edgecolor="white", linewidth=0.4)
    ref = [MANUSCRIPT_TABLE_TPR[m] for m in MODE_ORDER]
    ax.scatter(x, ref, marker="D", s=22, color="black", zorder=6,
               label="Manuscript Table 6.2 (quantum, ref.)")
    ax.set_xticks(x)
    ax.set_xticklabels([sl.FAILURE_MODE_LABELS[m].replace(" ", "\n") for m in MODE_ORDER],
                        fontsize=5.6)
    ax.set_ylabel("True positive rate (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="lower left", fontsize=5.6, ncol=1)


def _panel_c_auc_slope(ax, perf: pd.DataFrame) -> None:
    ordered = perf.set_index("failure_mode").loc[MODE_ORDER].reset_index()
    handles = []
    for i, row in ordered.iterrows():
        color = sl.FAILURE_MODE_COLORS[row.failure_mode]
        ax.plot([0, 1], [row.AUC_quantum, row.AUC_classical], color=color,
                lw=1.3, marker="o", markersize=4.2, zorder=5)
        handles.append(Line2D([0], [0], color=color, lw=1.5, marker="o", markersize=4.2,
                               label=sl.FAILURE_MODE_LABELS[row.failure_mode]))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Quantum", "Classical"])
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylabel("AUC (per-mode, vs. healthy)")
    ax.set_ylim(0.65, 1.05)
    ax.legend(handles=handles, loc="lower center", fontsize=5.2, ncol=1,
              bbox_to_anchor=(0.5, -0.02), framealpha=0.95)


def _panel_de_violin(ax, scores_df: pd.DataFrame, score_col: str, title: str, log_y: bool) -> None:
    groups = ["healthy"] + MODE_ORDER
    data = [np.clip(scores_df[scores_df.failure_mode == g][score_col].to_numpy(), 1e-6, None)
            for g in groups]
    if log_y:
        data_plot = [np.log10(d) for d in data]
    else:
        data_plot = data
    parts = ax.violinplot(data_plot, showmedians=True, widths=0.8)
    colors = [sl.ROLE_COLOR["healthy"]] + [sl.FAILURE_MODE_COLORS[m] for m in MODE_ORDER]
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.55); pc.set_edgecolor(c)
    for key in ["cmedians", "cmins", "cmaxes", "cbars"]:
        parts[key].set_color("0.25"); parts[key].set_linewidth(0.7)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(["Healthy"] + [sl.FAILURE_MODE_LABELS[m].replace(" ", "\n") for m in MODE_ORDER],
                        fontsize=5.4)
    ax.set_ylabel(("log$_{10}$(" + title + ")") if log_y else title)


def _panel_f_early_warning(ax, ew_df: pd.DataFrame) -> None:
    ordered = ew_df.set_index("failure_mode").loc[MODE_ORDER].reset_index()
    x = np.arange(len(MODE_ORDER)); w = 0.36
    ax.bar(x - w / 2, ordered.EW_quantum_hours_mean, width=w, color=sl.ROLE_COLOR["quantum"],
           yerr=ordered.EW_quantum_hours_std, capsize=2.2, edgecolor="white", linewidth=0.4,
           label="Quantum")
    ax.bar(x + w / 2, ordered.EW_classical_hours_mean, width=w, color=sl.ROLE_COLOR["classical"],
           yerr=ordered.EW_classical_hours_std, capsize=2.2, edgecolor="white", linewidth=0.4,
           label="Classical")
    ax.set_xticks(x)
    ax.set_xticklabels([sl.FAILURE_MODE_LABELS[m].replace(" ", "\n") for m in MODE_ORDER], fontsize=5.6)
    ax.set_ylabel("Early-warning horizon (hours)")
    ax.legend(loc="upper right", fontsize=6.2)


def build_figure(bundle) -> str:
    sl.print_section_header("FIGURE 2: Quantum vs. Classical Anomaly Detection Performance")
    scores = bundle.anomaly_scores
    roc = bundle.roc_curves
    perf = bundle.perf_by_mode
    ew = bundle.early_warning

    sl.print_numeric_summary(perf, label="Per-mode TPR/AUC/FPR (operating threshold)")
    sl.print_numeric_summary(ew, label="Early-warning horizon (hours)")
    overall_auc_q = np.trapezoid(roc[roc.detector == "quantum"].sort_values("fpr").tpr,
                                  roc[roc.detector == "quantum"].sort_values("fpr").fpr)
    overall_auc_c = np.trapezoid(roc[roc.detector == "classical"].sort_values("fpr").tpr,
                                  roc[roc.detector == "classical"].sort_values("fpr").fpr)
    print(f"  Overall AUC -- quantum: {overall_auc_q:.5f}   classical: {overall_auc_c:.5f}")

    fig = plt.figure(figsize=sl.figsize_grid(3, 2, panel_w=2.35, panel_h=2.05))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.48, hspace=0.62,
                            left=0.075, right=0.98, top=0.89, bottom=0.16)

    ax_a = fig.add_subplot(gs[0, 0]); _panel_a_roc(ax_a, roc, perf); sl.label_panel(ax_a, "a")
    ax_b = fig.add_subplot(gs[0, 1]); _panel_b_tpr_bars(ax_b, perf); sl.label_panel(ax_b, "b")
    ax_c = fig.add_subplot(gs[0, 2]); _panel_c_auc_slope(ax_c, perf); sl.label_panel(ax_c, "c")
    ax_d = fig.add_subplot(gs[1, 0])
    _panel_de_violin(ax_d, scores, "quantum_relentropy_score", "quantum score", log_y=True)
    sl.label_panel(ax_d, "d")
    ax_e = fig.add_subplot(gs[1, 1])
    _panel_de_violin(ax_e, scores, "classical_ocsvm_score", "classical score", log_y=False)
    sl.label_panel(ax_e, "e")
    ax_f = fig.add_subplot(gs[1, 2]); _panel_f_early_warning(ax_f, ew); sl.label_panel(ax_f, "f")

    fig.suptitle("Figure 2. Quantum vs. classical anomaly-detection performance across five failure modes",
                  x=0.075, ha="left", fontsize=10.2, fontweight="bold", y=0.995)

    n_pts = len(scores) + len(roc)
    path = sl.save_publication_figure(
        fig, "Figure_02_anomaly_detection_performance",
        "Quantum vs. classical anomaly-detection performance",
        n_panels=6, n_data_points=n_pts,
        notes="a: ROC (eq:anomaly_score_centred detector); b: TPR bars w/ manuscript "
              "Table 6.2 reference; c: per-mode AUC; d-e: score distributions; f: EW horizon.")
    plt.close(fig)
    return path


if __name__ == "__main__":
    import module_01_data_loader as dl
    bundle = dl.load_all_datasets()
    build_figure(bundle)
