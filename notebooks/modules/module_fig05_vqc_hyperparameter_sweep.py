from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(__file__))
import module_00_style_library as sl  # noqa: E402


def _panel_a_vs_L(ax, sweep: pd.DataFrame) -> None:
    sub = sweep[(sweep.n == 8) & (np.isclose(sweep.lr, 0.01))].sort_values("L")
    ax.errorbar(sub.L, sub.AUC_mean, yerr=sub.AUC_std, color=sl.OKABE_ITO["blue"],
                marker="o", markersize=5, lw=1.4, capsize=3)
    ax.set_xlabel("Circuit depth $L$ (layers)")
    ax.set_ylabel("AUC (validation)")
    ax.set_xticks(sub.L)
    ax.set_ylim(0.85, 1.02)


def _panel_b_vs_n(ax, sweep: pd.DataFrame) -> None:
    sub = sweep[(sweep.L == 4) & (np.isclose(sweep.lr, 0.01))].sort_values("n")
    ax.errorbar(sub.n, sub.AUC_mean, yerr=sub.AUC_std, color=sl.OKABE_ITO["bluish_green"],
                marker="s", markersize=5, lw=1.4, capsize=3)
    ax.set_xlabel("Number of qubits $n$")
    ax.set_ylabel("AUC (validation)")
    ax.set_xticks(sub.n)
    ax.set_ylim(0.85, 1.02)


def _panel_c_vs_lr(ax, sweep: pd.DataFrame) -> None:
    sub = sweep[(sweep.L == 4) & (sweep.n == 8)].sort_values("lr")
    ax.errorbar(sub.lr, sub.AUC_mean, yerr=sub.AUC_std, color=sl.OKABE_ITO["vermillion"],
                marker="^", markersize=5.5, lw=1.4, capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("AUC (validation)")
    ax.set_ylim(0.85, 1.02)


def _panel_d_parallel_coords(ax, sweep: pd.DataFrame) -> None:
    dims = ["L", "n", "lr", "AUC_mean", "FAR_at_95pct_thresh_mean"]
    dim_labels = ["Depth $L$", "Qubits $n$", "Learning\nrate", "AUC", "FAR\n(95th pct.)"]
    norm_data = {}
    for d in dims:
        v = sweep[d].to_numpy(dtype=float)
        if d == "lr":
            v = np.log10(v)
        vmin, vmax = v.min(), v.max()
        span = vmax - vmin if vmax > vmin else 1.0
        norm_data[d] = (v - vmin) / span
    x = np.arange(len(dims))
    cmap = plt.cm.get_cmap(sl.CMAP_SEQUENTIAL)
    auc_vals = sweep["AUC_mean"].to_numpy()
    colors = cmap((auc_vals - auc_vals.min()) / max(auc_vals.max() - auc_vals.min(), 1e-9))
    for i, (_, row) in enumerate(sweep.iterrows()):
        y = [norm_data[d][i] for d in dims]
        ax.plot(x, y, color=colors[i], lw=1.5, alpha=0.85,
                marker="o", markersize=3.2)
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=6.2)
    for xi in x:
        ax.axvline(xi, color="0.85", lw=0.6, zorder=0)
    ax.set_yticks([])
    ax.set_ylabel("Normalized range (per-axis min-max)")
    for d, xi in zip(dims, x):
        vals = sweep[d].to_numpy(dtype=float)
        lo, hi = vals.min(), vals.max()
        fmt = (lambda v: f"{v:.4g}") if d != "lr" else (lambda v: f"{v:.3g}")
        ax.text(xi, -0.08, fmt(lo), ha="center", va="top", fontsize=5.2, transform=ax.get_xaxis_transform())
        ax.text(xi, 1.03, fmt(hi), ha="center", va="bottom", fontsize=5.2, transform=ax.get_xaxis_transform())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(auc_vals.min(), auc_vals.max()))
    cb = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label("AUC (colour)", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5)


def build_figure(bundle) -> str:
    sl.print_section_header("FIGURE 5: VQC Hyperparameter Sweep (real parameter-shift training)")
    sweep = bundle.vqc_sweep
    sl.print_numeric_summary(sweep, columns=["L", "n", "lr", "AUC_mean", "AUC_std",
                                              "FAR_at_95pct_thresh_mean"],
                              label="VQC hyperparameter sweep (8 configs x 3 seeds)")
    print(sweep.to_string(index=False))

    fig = plt.figure(figsize=sl.figsize_grid(2, 2, panel_w=2.9, panel_h=2.25))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.36, hspace=0.55,
                            left=0.09, right=0.94, top=0.89, bottom=0.14)

    ax_a = fig.add_subplot(gs[0, 0]); _panel_a_vs_L(ax_a, sweep); sl.label_panel(ax_a, "a")
    ax_b = fig.add_subplot(gs[0, 1]); _panel_b_vs_n(ax_b, sweep); sl.label_panel(ax_b, "b")
    ax_c = fig.add_subplot(gs[1, 0]); _panel_c_vs_lr(ax_c, sweep); sl.label_panel(ax_c, "c")
    ax_d = fig.add_subplot(gs[1, 1]); _panel_d_parallel_coords(ax_d, sweep); sl.label_panel(ax_d, "d")

    fig.suptitle("Figure 5. VQC hyperparameter sweep -- real parameter-shift-rule training results",
                  x=0.09, ha="left", fontsize=10.2, fontweight="bold", y=0.995)

    path = sl.save_publication_figure(
        fig, "Figure_05_vqc_hyperparameter_sweep", "VQC hyperparameter sweep",
        n_panels=4, n_data_points=len(sweep),
        notes="Real gradient-descent training via exact parameter-shift rule (eq:parameter_shift_rule); "
              "8 configs x 3 seeds, reduced from manuscript's 10 seeds for compute budget.")
    plt.close(fig)
    return path


if __name__ == "__main__":
    import module_01_data_loader as dl
    bundle = dl.load_all_datasets()
    build_figure(bundle)
