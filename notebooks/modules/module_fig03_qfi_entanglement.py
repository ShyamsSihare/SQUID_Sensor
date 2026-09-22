from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(__file__))
import module_00_style_library as sl  # noqa: E402

YEARS_SNAPSHOTS = [0, 1, 2, 3, 5, 7, 10]
SNAPSHOT_COLORS = plt.cm.get_cmap(sl.CMAP_SEQUENTIAL)(np.linspace(0.05, 0.92, len(YEARS_SNAPSHOTS)))


def _panel_a_qfi_vs_N(ax, qfi_df: pd.DataFrame) -> None:
    for yr, color in zip(YEARS_SNAPSHOTS, SNAPSHOT_COLORS):
        sub = qfi_df[qfi_df.years == yr].sort_values("N")
        ax.plot(sub.N, sub.QFI, color=color, lw=1.3, marker="o", markersize=3,
                label=f"{yr} yr")
    N_ref = np.array(sorted(qfi_df.N.unique()))
    ax.plot(N_ref, N_ref.astype(float) ** 2, color="black", lw=1.0, ls="--",
            label=r"Heisenberg limit $N^2$")
    ax.plot(N_ref, N_ref.astype(float), color="0.5", lw=1.0, ls=":",
            label=r"Standard quantum limit $N$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Array size $N$")
    ax.set_ylabel("Quantum Fisher information")
    ax.legend(loc="upper left", fontsize=5.0, ncol=1)


def _panel_b_surface(fig, ax, qfi_df: pd.DataFrame) -> None:
    Ns = np.array(sorted(qfi_df.N.unique()))
    yrs = np.array(sorted(qfi_df.years.unique()))
    pivot = qfi_df.pivot_table(index="years", columns="N", values="QFI").reindex(
        index=yrs, columns=Ns)
    Xg, Yg = np.meshgrid(np.log10(Ns), yrs)
    Zg = np.log10(pivot.to_numpy())
    surf = ax.plot_surface(Xg, Yg, Zg, cmap=sl.CMAP_SEQUENTIAL, linewidth=0,
                            antialiased=True, alpha=0.95, rasterized=True)
    ax.set_xlabel(r"$\log_{10} N$", fontsize=6.5, labelpad=1)
    ax.set_ylabel("years", fontsize=6.5, labelpad=1)
    ax.tick_params(labelsize=5.2, pad=0)
    ax.view_init(elev=22, azim=-58)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.02, label=r"$\log_{10}(\mathrm{QFI})$")


def _panel_c_visibility(ax, qfi_df: pd.DataFrame) -> None:
    sub = qfi_df[qfi_df.N == 64].sort_values("years")
    ax.plot(sub.years, sub.visibility, color=sl.OKABE_ITO["blue"], lw=1.6, marker="o", markersize=3.4)
    ax.axhline(1 / np.e, color="0.6", ls="--", lw=0.8)
    ax.text(0.3, 1 / np.e + 0.02, r"$1/e$", fontsize=6.5, color="0.4")
    ax.set_xlabel("Operating time (years)")
    ax.set_ylabel(r"Entanglement visibility $\mathcal{V}(\tau)$")
    ax.set_ylim(0, 1.05)


def _panel_d_qfi_degradation(ax, qfi_df: pd.DataFrame) -> None:
    sub = qfi_df[qfi_df.N == 64].sort_values("years")
    qfi_fresh = sub.QFI.iloc[0]
    ax.plot(sub.years, sub.QFI, color=sl.OKABE_ITO["reddish_purple"], lw=1.6, marker="s",
            markersize=3.6, label="GHZ, $N{=}64$ (dephased, this work)")
    ax.axhline(qfi_fresh, color="black", ls="--", lw=0.9, label=r"Heisenberg ceiling $N^2$ (no decoherence)")
    ax.axhline(64, color="0.55", ls=":", lw=0.9, label="Standard quantum limit $N$")
    ax.set_yscale("log")
    ax.set_xlabel("Operating time (years)")
    ax.set_ylabel("QFI ($N=64$)")
    ax.legend(loc="lower left", fontsize=5.4)


def _panel_e_optimal_N(ax, lifecycle: pd.DataFrame) -> None:
    Nstar = 1.0 / (lifecycle["Gamma_deph_per_s"] * (8.0e-3 / (64 * 1.26e5)))
    ax.plot(lifecycle.years, Nstar, color=sl.OKABE_ITO["bluish_green"], lw=1.6)
    ax.axhline(64, color="0.5", ls="--", lw=0.8, label="Deployed array size $N{=}64$")
    ax.set_xlabel("Operating time (years)")
    ax.set_ylabel(r"Optimal array size $N^\star(t)$")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=6.0)


def _panel_f_contour(fig, ax, qfi_df: pd.DataFrame) -> None:
    Ns = np.array(sorted(qfi_df.N.unique()))
    yrs = np.array(sorted(qfi_df.years.unique()))
    pivot = qfi_df.pivot_table(index="years", columns="N", values="QFI").reindex(index=yrs, columns=Ns)
    Xg, Yg = np.meshgrid(Ns, yrs)
    cs = ax.contourf(Xg, Yg, pivot.to_numpy(), levels=30, cmap=sl.CMAP_SEQUENTIAL,
                      norm=LogNorm(vmin=pivot.to_numpy().min() + 1e-3, vmax=pivot.to_numpy().max()))
    ax.set_xscale("log")
    ax.set_xlabel("Array size $N$")
    ax.set_ylabel("Operating time (years)")
    sl.styled_colorbar(cs, ax, "QFI", fig)


def build_figure(bundle) -> str:
    sl.print_section_header("FIGURE 3: Quantum Fisher Information & Entanglement-Enhanced Sensing")
    qfi_df = bundle.qfi_vs_N_dose
    lifecycle = bundle.lifecycle
    sl.print_numeric_summary(qfi_df, columns=["N", "years", "QFI", "visibility"],
                              label="QFI vs N and dose")

    fig = plt.figure(figsize=sl.figsize_grid(3, 2, panel_w=2.35, panel_h=2.1))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.50, hspace=0.52,
                            left=0.07, right=0.98, top=0.90, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0]); _panel_a_qfi_vs_N(ax_a, qfi_df); sl.label_panel(ax_a, "a")
    ax_b = fig.add_subplot(gs[0, 1], projection="3d"); _panel_b_surface(fig, ax_b, qfi_df)
    ax_b.text2D(-0.05, 1.0, "b", transform=ax_b.transAxes, fontsize=plt.rcParams["font.size"] + 2,
                fontweight="bold", va="bottom", ha="right")
    ax_c = fig.add_subplot(gs[0, 2]); _panel_c_visibility(ax_c, qfi_df); sl.label_panel(ax_c, "c")
    ax_d = fig.add_subplot(gs[1, 0]); _panel_d_qfi_degradation(ax_d, qfi_df); sl.label_panel(ax_d, "d")
    ax_e = fig.add_subplot(gs[1, 1]); _panel_e_optimal_N(ax_e, lifecycle); sl.label_panel(ax_e, "e")
    ax_f = fig.add_subplot(gs[1, 2]); _panel_f_contour(fig, ax_f, qfi_df); sl.label_panel(ax_f, "f")

    fig.suptitle("Figure 3. Quantum Fisher information and entanglement-enhanced sensing vs. array size and dose",
                  x=0.07, ha="left", fontsize=10.0, fontweight="bold", y=0.995)

    path = sl.save_publication_figure(
        fig, "Figure_03_qfi_entanglement", "QFI and entanglement-enhanced sensing",
        n_panels=6, n_data_points=len(qfi_df) + len(lifecycle),
        notes="Exact closed form eq:exact_qfi_dephased, QFI=N^2 exp(-2N*Gamma_deph*tau).")
    plt.close(fig)
    return path


if __name__ == "__main__":
    import module_01_data_loader as dl
    bundle = dl.load_all_datasets()
    build_figure(bundle)
