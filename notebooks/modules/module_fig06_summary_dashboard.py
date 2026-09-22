from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(__file__))
import module_00_style_library as sl  # noqa: E402


def _panel_a_spectra_thumb(ax, noise_df: pd.DataFrame) -> None:
    for stage, color, lbl in [("fresh", sl.ROLE_COLOR["fresh"], "Fresh"),
                               ("7yr_prefail", sl.ROLE_COLOR["stage4"], "Pre-failure")]:
        g = noise_df[noise_df.stage == stage].groupby("freq_hz")["sqrt_S_phi_Phi0_per_sqrtHz"].mean() * 1e6
        ax.plot(g.index, g.to_numpy(), color=color, lw=1.6, label=lbl)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$f$ (Hz)"); ax.set_ylabel(r"$\sqrt{S_\Phi}$ ($\mu\Phi_0/\sqrt{\rm Hz}$)")
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title("Noise-floor growth", fontsize=7.5, loc="center")


def _panel_b_qfi_thumb(ax, qfi_df: pd.DataFrame) -> None:
    sub = qfi_df[qfi_df.N == 64].sort_values("years")
    ax.plot(sub.years, sub.QFI, color=sl.OKABE_ITO["reddish_purple"], lw=1.8, marker="o", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("Operating time (yr)"); ax.set_ylabel("QFI ($N{=}64$)")
    ax.set_title("Entanglement-sensing degradation", fontsize=7.5, loc="center")


def _panel_c_radar(ax, perf_by_mode: pd.DataFrame, roc_df: pd.DataFrame,
                    ew_df: pd.DataFrame, sweep: pd.DataFrame) -> None:
    auc_q = np.trapezoid(roc_df[roc_df.detector == "quantum"].sort_values("fpr").tpr,
                          roc_df[roc_df.detector == "quantum"].sort_values("fpr").fpr)
    auc_c = np.trapezoid(roc_df[roc_df.detector == "classical"].sort_values("fpr").tpr,
                          roc_df[roc_df.detector == "classical"].sort_values("fpr").fpr)
    metrics = ["Overall\nAUC", "Mean TPR\n(%)/100", "Mean EW\n(hrs)/60", "VQC best\nAUC", "1 - FPR\n(x1e-4/100)"]
    q_vals = [auc_q, perf_by_mode.TPR_quantum_pct.mean() / 100,
              ew_df.EW_quantum_hours_mean.mean() / 60, sweep.AUC_mean.max(),
              1 - perf_by_mode["FPR_quantum_x1e-4"].mean() / 100]
    c_vals = [auc_c, perf_by_mode.TPR_classical_pct.mean() / 100,
              ew_df.EW_classical_hours_mean.mean() / 60, sweep.AUC_mean.max(),
              1 - perf_by_mode["FPR_classical_x1e-4"].mean() / 100]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    q_vals += q_vals[:1]; c_vals += c_vals[:1]
    ax.plot(angles, q_vals, color=sl.ROLE_COLOR["quantum"], lw=1.6, marker="o", markersize=3.5,
            label="Quantum")
    ax.fill(angles, q_vals, color=sl.ROLE_COLOR["quantum"], alpha=0.15)
    ax.plot(angles, c_vals, color=sl.ROLE_COLOR["classical"], lw=1.6, marker="s", markersize=3.5,
            label="Classical")
    ax.fill(angles, c_vals, color=sl.ROLE_COLOR["classical"], alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=5.6)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=6.0, bbox_to_anchor=(1.25, 1.15))
    ax.set_title("Multi-metric comparison", fontsize=7.5, y=1.12)


def _panel_d_qpms_thumb(ax, qpms: pd.DataFrame) -> None:
    ax.plot(qpms.year, qpms.qpms_scalar, color="black", lw=1.6)
    ax.axhline(0.75, color=sl.OKABE_ITO["vermillion"], ls="--", lw=0.9)
    cross = qpms[qpms.qpms_scalar >= 0.75].year.min()
    if pd.notna(cross):
        ax.axvline(cross, color=sl.OKABE_ITO["vermillion"], ls=":", lw=0.8)
        ax.annotate(f"crit. @ {cross:.2f} yr", xy=(cross, 0.75), xytext=(cross - 1.6, 0.9),
                    fontsize=6.0, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("Operating time (yr)"); ax.set_ylabel("QPMS")
    ax.set_title("Predictive-maintenance trajectory", fontsize=7.5, loc="center")


def _panel_e_scorecard(ax, bundle) -> None:
    ax.axis("off")
    roc_df = bundle.roc_curves
    auc_q = np.trapezoid(roc_df[roc_df.detector == "quantum"].sort_values("fpr").tpr,
                          roc_df[roc_df.detector == "quantum"].sort_values("fpr").fpr)
    lifecycle = bundle.lifecycle
    qfi0 = lifecycle.QFI_N64.iloc[0]
    qfi10 = lifecycle.QFI_N64.iloc[-1]
    lines = [
        ("Overall quantum-detector AUC", f"{auc_q:.3f}"),
        ("Best VQC sweep AUC", f"{bundle.vqc_sweep.AUC_mean.max():.3f}"),
        ("QFI retained at 10 yr (N=64)", f"{100*qfi10/qfi0:.1f}%"),
        ("Ic retained at 10 yr", f"{100*lifecycle.Ic_uA.iloc[-1]/lifecycle.Ic_uA.iloc[0]:.1f}%"),
        ("Mean early-warning horizon", f"{bundle.early_warning.EW_quantum_hours_mean.mean():.1f} h"),
        ("Hardest failure mode", "Coolant flow anomaly"),
        ("Simulated test spectra", f"{len(bundle.anomaly_scores):,}"),
        ("Total simulated data points", f"{sum(len(getattr(bundle, f)) for f in bundle.__dataclass_fields__):,}"),
    ]
    y0 = 0.94
    ax.text(0.02, 1.02, "Key simulated results", fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    for i, (k, v) in enumerate(lines):
        y = y0 - i * 0.125
        ax.text(0.02, y, k, fontsize=6.6, transform=ax.transAxes, va="top")
        ax.text(0.98, y, v, fontsize=6.6, transform=ax.transAxes, va="top", ha="right", fontweight="bold",
                color=sl.OKABE_ITO["blue"])
        if i < len(lines) - 1:
            ax.plot([0, 1], [y - 0.045, y - 0.045], color="0.88", lw=0.5, transform=ax.transAxes)


def build_figure(bundle) -> str:
    sl.print_section_header("FIGURE 6: Graphical-Abstract-Style Summary Dashboard")

    fig = plt.figure(figsize=(sl.COLUMN_WIDTH_DOUBLE_IN, 5.6))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.55, hspace=0.55,
                            left=0.07, right=0.96, top=0.89, bottom=0.10,
                            width_ratios=[1, 1, 1.15])

    ax_a = fig.add_subplot(gs[0, 0]); _panel_a_spectra_thumb(ax_a, bundle.noise_spectra); sl.label_panel(ax_a, "a")
    ax_b = fig.add_subplot(gs[0, 1]); _panel_b_qfi_thumb(ax_b, bundle.qfi_vs_N_dose); sl.label_panel(ax_b, "b")
    ax_c = fig.add_subplot(gs[0, 2], projection="polar")
    _panel_c_radar(ax_c, bundle.perf_by_mode, bundle.roc_curves, bundle.early_warning, bundle.vqc_sweep)
    ax_d = fig.add_subplot(gs[1, 0]); _panel_d_qpms_thumb(ax_d, bundle.qpms_trajectory); sl.label_panel(ax_d, "d")
    ax_e = fig.add_subplot(gs[1, 1:]); _panel_e_scorecard(ax_e, bundle); sl.label_panel(ax_e, "e", x=-0.02, y=1.02)

    fig.text(0.02, 0.97, "c", fontsize=plt.rcParams["font.size"] + 2, fontweight="bold")

    fig.suptitle("Figure 6. Graphical-abstract summary of the simulated numerical experiments",
                  x=0.07, ha="left", fontsize=10.2, fontweight="bold", y=0.995)

    n_pts = sum(len(getattr(bundle, f)) for f in bundle.__dataclass_fields__)
    path = sl.save_publication_figure(
        fig, "Figure_06_summary_dashboard", "Graphical-abstract summary dashboard",
        n_panels=5, n_data_points=n_pts,
        notes="Headline results drawn from Figures 1-5; radar chart is a multi-metric "
              "quantum-vs-classical comparison across AUC/TPR/EW/FPR.")
    plt.close(fig)
    return path


if __name__ == "__main__":
    import module_01_data_loader as dl
    bundle = dl.load_all_datasets()
    build_figure(bundle)
