from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy import interpolate as _interp

sys.path.insert(0, os.path.dirname(__file__))
import module_00_style_library as sl  # noqa: E402

STAGE_ORDER = ["fresh", "2yr", "5yr_precursor", "7yr_prefail"]
STAGE_YEARS = {"fresh": 0, "2yr": 2, "5yr_precursor": 5, "7yr_prefail": 7}
STAGE_LABELS = {
    "fresh": "Stage 1: Fresh (0 yr)",
    "2yr": "Stage 2: 2 yr",
    "5yr_precursor": "Stage 3: 5 yr (precursor)",
    "7yr_prefail": "Stage 4: Pre-failure (7 yr, calibration ref.)",
}
STAGE_COLORS = {
    "fresh": sl.ROLE_COLOR["fresh"],
    "2yr": sl.ROLE_COLOR["stage2"],
    "5yr_precursor": sl.ROLE_COLOR["stage3"],
    "7yr_prefail": sl.ROLE_COLOR["stage4"],
}


def _panel_a_spectra(ax, noise_df: pd.DataFrame) -> None:
    """Log-log noise spectra, mean +/- 1 sigma band, four lifecycle stages."""
    for stage in STAGE_ORDER:
        sub = noise_df[noise_df.stage == stage]
        g = sub.groupby("freq_hz")["sqrt_S_phi_Phi0_per_sqrtHz"].agg(["mean", "std"])
        g = g.sort_index()
        f = g.index.to_numpy()
        mean_uphi = g["mean"].to_numpy() * 1e6
        std_uphi = g["std"].to_numpy() * 1e6
        c = STAGE_COLORS[stage]
        ax.plot(f, mean_uphi, color=c, lw=1.4, label=STAGE_LABELS[stage], zorder=5)
        ax.fill_between(f, np.clip(mean_uphi - std_uphi, 1e-6, None), mean_uphi + std_uphi,
                         color=c, alpha=0.18, linewidth=0, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Frequency $f$ (Hz)")
    ax.set_ylabel(r"$\sqrt{S_\Phi}$ ($\mu\Phi_0/\sqrt{\mathrm{Hz}}$)")
    ax.set_xlim(1e-2, 1e4)
    leg = ax.legend(loc="lower left", fontsize=6.2, ncol=1)
    leg.get_frame().set_linewidth(0.5)
    # 1/f^0.5 reference guide (since sqrt(S) ~ sqrt(1/f) = f^-0.5)
    f_guide = np.logspace(-2, 1.3, 20)
    guide = mean_uphi[0] * (f_guide / f[0]) ** (-0.5) * 0.9
    ax.plot(f_guide, guide, ls=":", color="0.4", lw=1.0, zorder=1)
    ax.text(0.25, guide[3] * 1.3, r"$f^{-1/2}$", fontsize=6.5, color="0.35")


def _panel_b_inset(ax_parent, noise_df: pd.DataFrame) -> None:
    axins = sl.inset_zoom(ax_parent, bounds=[0.50, 0.55, 0.46, 0.40],
                           xlim=(1e-2, 1.0), ylim=(1e2, 2e5))
    for stage in STAGE_ORDER:
        sub = noise_df[noise_df.stage == stage]
        g = sub.groupby("freq_hz")["sqrt_S_phi_Phi0_per_sqrtHz"].mean().sort_index() * 1e6
        axins.plot(g.index, g.to_numpy(), color=STAGE_COLORS[stage], lw=1.1)
    axins.set_xscale("log"); axins.set_yscale("log")


def _panel_c_corner_freq(ax, lifecycle: pd.DataFrame) -> None:
    ax.plot(lifecycle.years, lifecycle["corner_freq_Hz"], color=sl.OKABE_ITO["black"], lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Operating time (years)")
    ax.set_ylabel(r"Corner frequency $f_c$ (Hz)")
    for stage, yrs in STAGE_YEARS.items():
        idx = (lifecycle.years - yrs).abs().idxmin()
        ax.scatter([yrs], [lifecycle.loc[idx, "corner_freq_Hz"]],
                   color=STAGE_COLORS[stage], s=32, zorder=6,
                   edgecolor="white", linewidth=0.6)
    ax.set_xlim(0, 10)


def _panel_d_critical_current(ax, lifecycle: pd.DataFrame) -> None:
    ratio = lifecycle["Ic_uA"] / lifecycle["Ic_uA"].iloc[0]
    ax.plot(lifecycle.years, ratio * 100, color=sl.OKABE_ITO["vermillion"], lw=1.6)
    ax.axhline(100, color="0.7", lw=0.6, ls="--")
    ax.set_xlabel("Operating time (years)")
    ax.set_ylabel(r"$I_c(t)/I_c(0)$ (%)")
    ax.set_xlim(0, 10)
    ax.fill_between(lifecycle.years, ratio * 100, 100, color=sl.OKABE_ITO["vermillion"], alpha=0.10)
    end_drop = 100 - ratio.iloc[-1] * 100
    ax.annotate(f"-{end_drop:.1f}% @ 10 yr", xy=(9.7, ratio.iloc[-1] * 100),
                xytext=(4.2, 92.5), fontsize=6.4,
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.3"))


def _panel_e_A_phi_growth(ax, lifecycle: pd.DataFrame) -> None:
    ax.plot(lifecycle["ddd_dpa"], lifecycle["A_phi_Phi0^2_per_Hz"] * 1e6,
            color=sl.OKABE_ITO["bluish_green"], lw=1.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Displacement damage dose (dpa)")
    ax.set_ylabel(r"$A_\varphi(t)$ ($\mu\Phi_0^2/\mathrm{Hz}$)")
    for stage, yrs in STAGE_YEARS.items():
        idx = (lifecycle.years - yrs).abs().idxmin()
        ax.scatter([lifecycle.loc[idx, "ddd_dpa"]], [lifecycle.loc[idx, "A_phi_Phi0^2_per_Hz"] * 1e6],
                   color=STAGE_COLORS[stage], s=32, zorder=6, edgecolor="white", linewidth=0.6)


def _panel_f_spectrogram(fig, ax, lifecycle: pd.DataFrame, noise_df: pd.DataFrame) -> None:
    freqs = np.sort(noise_df.freq_hz.unique())
    years = np.linspace(0, 10, 120)
    A_phi_interp = _interp.interp1d(lifecycle.years, lifecycle["A_phi_Phi0^2_per_Hz"],
                                     kind="linear", fill_value="extrapolate")
    thermal_floor = noise_df.groupby("stage")["S_phi_Phi0^2_per_Hz"].min().mean() * 0.0 + \
        (noise_df[noise_df.stage == "fresh"]["S_phi_Phi0^2_per_Hz"].to_numpy()[-1])
    Z = np.zeros((len(years), len(freqs)))
    for i, yr in enumerate(years):
        A = A_phi_interp(yr)
        Z[i, :] = A / freqs + thermal_floor
    Z_fresh = Z[0, :]
    Z_ratio = Z / Z_fresh[None, :]  # damage-induced amplification relative to fresh spectrum
    pcm = ax.pcolormesh(freqs, years, Z_ratio, cmap=sl.CMAP_DAMAGE,
                         norm=LogNorm(vmin=1.0, vmax=max(Z_ratio.max(), 1.01)),
                         shading="auto", rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel(r"Frequency $f$ (Hz)")
    ax.set_ylabel("Operating time (years)")
    for stage, yrs in STAGE_YEARS.items():
        ax.axhline(yrs, color="white", lw=0.5, ls=":", alpha=0.7)
    cb = sl.styled_colorbar(pcm, ax, r"$S_\Phi(f,t)\,/\,S_\Phi(f,0)$ (damage factor)", fig)


def build_figure(bundle) -> str:
    sl.print_section_header("FIGURE 1: SQUID Noise-Spectrum Evolution Across Sensor Lifetime")
    noise_df = bundle.noise_spectra
    lifecycle = bundle.lifecycle

    sl.print_numeric_summary(
        noise_df.assign(sqrt_S_uphi=noise_df["sqrt_S_phi_Phi0_per_sqrtHz"] * 1e6),
        columns=["sqrt_S_uphi"], label="sqrt(S_phi) across all stages/freqs [uPhi0/sqrtHz]")
    sl.print_numeric_summary(
        lifecycle, columns=["ddd_dpa", "Ic_uA", "corner_freq_Hz", "QFI_N64"],
        label="Lifecycle trace (0-10 yr, 200 samples)")

    fig = plt.figure(figsize=sl.figsize_grid(3, 2, panel_w=2.35, panel_h=2.0))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.42, hspace=0.48,
                            left=0.08, right=0.98, top=0.89, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0:2])
    _panel_a_spectra(ax_a, noise_df)
    _panel_b_inset(ax_a, noise_df)
    sl.label_panel(ax_a, "a")

    ax_c = fig.add_subplot(gs[0, 2])
    _panel_c_corner_freq(ax_c, lifecycle)
    sl.label_panel(ax_c, "b")

    ax_d = fig.add_subplot(gs[1, 0])
    _panel_d_critical_current(ax_d, lifecycle)
    sl.label_panel(ax_d, "c")

    ax_e = fig.add_subplot(gs[1, 1])
    _panel_e_A_phi_growth(ax_e, lifecycle)
    sl.label_panel(ax_e, "d")

    ax_f = fig.add_subplot(gs[1, 2])
    _panel_f_spectrogram(fig, ax_f, lifecycle, noise_df)
    sl.label_panel(ax_f, "e")

    fig.suptitle("Figure 1. SQUID flux-noise spectrum evolution across sensor operating life",
                  x=0.08, ha="left", fontsize=10.5, fontweight="bold", y=0.995)

    n_pts = len(noise_df) + len(lifecycle)
    path = sl.save_publication_figure(
        fig, "Figure_01_noise_spectrum_evolution",
        "SQUID flux-noise spectrum evolution across sensor operating life",
        n_panels=5, n_data_points=n_pts,
        notes="Panels a-b: spectra (eq:total_noise); c: corner freq (eq:corner_frequency); "
              "d: Ic(t) (eq:critical_current_degradation); e: A_phi(t) (eq:tls_noise); "
              "f: continuous spectrogram interpolation.")
    plt.close(fig)
    return path


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    import module_01_data_loader as dl
    bundle = dl.load_all_datasets()
    build_figure(bundle)
