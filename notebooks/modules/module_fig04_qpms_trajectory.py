from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(__file__))
import module_00_style_library as sl  # noqa: E402

QPMS_CRIT = 0.75
QPMS_EW = 0.55


def _panel_a_stacked(ax, qpms: pd.DataFrame) -> None:
    """Individual weighted-channel trajectories (NOT stacked -- QPMS is a
    weighted AVERAGE of these three channels, so a stackplot would visually
    imply summation that doesn't match the underlying composite formula;
    plotting them as separate lines against the composite avoids that
    misleading impression)."""
    ax.plot(qpms.year, qpms.anomaly_channel, color=sl.OKABE_ITO["blue"], lw=1.1, alpha=0.85,
            label=r"Anomaly channel $\mathcal{A}(t)/\mathcal{A}_{\rm thresh}$")
    ax.plot(qpms.year, qpms.qfi_degradation_channel, color=sl.OKABE_ITO["bluish_green"], lw=1.1,
            alpha=0.85, label=r"QFI-degradation channel $1-\mathcal{Q}(t)/\mathcal{Q}(0)$")
    ax.plot(qpms.year, qpms.fluence_channel, color=sl.OKABE_ITO["orange"], lw=1.1, alpha=0.85,
            label=r"Fluence channel $\mathcal{D}(t)$")
    ax.plot(qpms.year, qpms.qpms_scalar, color="black", lw=1.9, label="Composite QPMS "
            r"$=w_A\mathcal{A}+w_Q(1-\mathcal{Q})+w_D\mathcal{D}$")
    ax.axhline(QPMS_CRIT, color=sl.OKABE_ITO["vermillion"], ls="--", lw=1.0,
               label=r"Critical threshold $\mathrm{QPMS}^{\rm crit}$")
    ax.axhline(QPMS_EW, color=sl.OKABE_ITO["blue"], ls=":", lw=1.0,
               label="Early-warning threshold")
    ax.set_xlabel("Operating time (years)")
    ax.set_ylabel("Channel value / composite QPMS")
    ax.set_xlim(0, 5); ax.set_ylim(0, 1.3)
    ax.legend(loc="upper left", fontsize=4.9, ncol=1)


def _panel_b_detection_prob(ax, detect_df: pd.DataFrame) -> None:
    for N in sorted(detect_df.N.unique()):
        sub = detect_df[detect_df.N == N].sort_values("hours_before_hard_alarm")
        color = {4: sl.OKABE_ITO["orange"], 16: sl.OKABE_ITO["bluish_green"],
                 64: sl.OKABE_ITO["blue"]}.get(N, "black")
        ax.plot(sub.hours_before_hard_alarm, sub.detection_probability, color=color, lw=1.6,
                label=f"$N={N}$")
        f = sub.set_index("hours_before_hard_alarm")["detection_probability"]
        cross_idx = (f - 0.5).abs().idxmin()
        ax.scatter([cross_idx], [0.5], color=color, s=26, zorder=6, edgecolor="white", linewidth=0.5)
    ax.axhline(0.5, color="0.7", lw=0.6, ls="--")
    ax.set_xlabel("Hours before hard classical alarm")
    ax.set_ylabel("Early-detection probability")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_ylim(0, 1.05)


def _panel_c_phase_portrait(fig, ax, qpms: pd.DataFrame) -> None:
    q = qpms.qpms_scalar.to_numpy()
    t = qpms.day.to_numpy()
    dqdt = np.gradient(q, t)
    sc = ax.scatter(q, dqdt, c=qpms.year, cmap=sl.CMAP_SEQUENTIAL, s=6, linewidth=0, rasterized=True)
    ax.axvline(QPMS_CRIT, color=sl.OKABE_ITO["vermillion"], ls="--", lw=0.8)
    ax.axvline(QPMS_EW, color=sl.OKABE_ITO["blue"], ls=":", lw=0.8)
    ax.set_xlabel("QPMS scalar")
    ax.set_ylabel(r"$d(\mathrm{QPMS})/dt$ (day$^{-1}$)")
    sl.styled_colorbar(sc, ax, "Operating time (years)", fig)


def _panel_d_cdf(ax, qpms: pd.DataFrame) -> None:
    q_sorted = np.sort(qpms.qpms_scalar.to_numpy())
    cdf = np.arange(1, len(q_sorted) + 1) / len(q_sorted)
    ax.plot(q_sorted, cdf, color="black", lw=1.5)
    ax.axvline(QPMS_EW, color=sl.OKABE_ITO["blue"], ls=":", lw=1.0, label="Early-warning threshold")
    ax.axvline(QPMS_CRIT, color=sl.OKABE_ITO["vermillion"], ls="--", lw=1.0, label="Critical threshold")
    frac_normal = (qpms.qpms_scalar < QPMS_EW).mean() * 100
    frac_warn = ((qpms.qpms_scalar >= QPMS_EW) & (qpms.qpms_scalar < QPMS_CRIT)).mean() * 100
    frac_crit = (qpms.qpms_scalar >= QPMS_CRIT).mean() * 100
    ax.fill_betweenx([0, 1], 0, QPMS_EW, color=sl.OKABE_ITO["bluish_green"], alpha=0.08)
    ax.fill_betweenx([0, 1], QPMS_EW, QPMS_CRIT, color=sl.OKABE_ITO["orange"], alpha=0.10)
    ax.fill_betweenx([0, 1], QPMS_CRIT, max(q_sorted.max(), QPMS_CRIT + 0.05),
                      color=sl.OKABE_ITO["vermillion"], alpha=0.10)
    ax.text(QPMS_EW / 2, 0.5, f"Normal\n{frac_normal:.0f}%", ha="center", fontsize=6.2)
    ax.text((QPMS_EW + QPMS_CRIT) / 2, 0.3, f"Watch\n{frac_warn:.0f}%", ha="center", fontsize=6.2)
    ax.set_xlabel("QPMS scalar")
    ax.set_ylabel("Cumulative fraction of 5-yr operation")
    ax.set_xlim(0, max(q_sorted.max() * 1.05, QPMS_CRIT + 0.05))
    ax.legend(loc="lower right", fontsize=6.0)


def build_figure(bundle) -> str:
    sl.print_section_header("FIGURE 4: Quantum Predictive Maintenance Score (QPMS) Trajectory")
    qpms = bundle.qpms_trajectory
    detect = bundle.detect_prob_vs_N
    sl.print_numeric_summary(qpms, columns=["anomaly_channel", "qfi_degradation_channel",
                                             "fluence_channel", "qpms_scalar"],
                              label="QPMS trajectory channels (5-yr simulation)")

    fig = plt.figure(figsize=sl.figsize_grid(2, 2, panel_w=2.9, panel_h=2.35))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.38, hspace=0.48,
                            left=0.09, right=0.96, top=0.90, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0]); _panel_a_stacked(ax_a, qpms); sl.label_panel(ax_a, "a")
    ax_b = fig.add_subplot(gs[0, 1]); _panel_b_detection_prob(ax_b, detect); sl.label_panel(ax_b, "b")
    ax_c = fig.add_subplot(gs[1, 0]); _panel_c_phase_portrait(fig, ax_c, qpms); sl.label_panel(ax_c, "c")
    ax_d = fig.add_subplot(gs[1, 1]); _panel_d_cdf(ax_d, qpms); sl.label_panel(ax_d, "d")

    fig.suptitle("Figure 4. Quantum predictive maintenance score (QPMS) trajectory over simulated 5-year operation",
                  x=0.09, ha="left", fontsize=10.2, fontweight="bold", y=0.995)

    path = sl.save_publication_figure(
        fig, "Figure_04_qpms_trajectory", "QPMS trajectory over 5-year operation",
        n_panels=4, n_data_points=len(qpms) + len(detect),
        notes="Three-channel QPMS composite (Sec. 5.1); detection-probability curves for N=4/16/64.")
    plt.close(fig)
    return path


if __name__ == "__main__":
    import module_01_data_loader as dl
    bundle = dl.load_all_datasets()
    build_figure(bundle)
