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

# ------------------------------------------------------------------------------
# A compact, self-contained re-implementation of the relevant physics.py
# functions, parameterised so every calibration/material constant can be
# perturbed for this sensitivity study without touching the main simulation
# package (keeps this figure module fully standalone and auditable).
# ------------------------------------------------------------------------------
kB = 1.380649e-23
SECONDS_PER_YEAR = 365.25 * 86400.0
Phi_n0 = 1.2e11
sigma_dpa = 600e-27
xi_arc = 0.30
T_REF = 7 * SECONDS_PER_YEAR
Phi_gamma0 = 4.5e12
N_ARRAY = 64
R_n = 5.0
L_sq = 80e-12
T_op = 4.2
PHI0 = 2.067833848e-15


def ddd(t):
    return xi_arc * Phi_n0 * sigma_dpa * t


DDD_REF = ddd(T_REF)


def damage_fraction(t):
    return ddd(t) / DDD_REF


def s_thermal_floor():
    return (16 * kB * T_op * L_sq ** 2 / R_n) / PHI0 ** 2


def A_phi_of_t(t, A_phi0, kappa_eff):
    return A_phi0 * np.sqrt(1 + kappa_eff * damage_fraction(t))


def corner_freq(t, A_phi0, kappa_eff):
    return A_phi_of_t(t, A_phi0, kappa_eff) / s_thermal_floor()


def fluence_fraction(t):
    return (Phi_gamma0 * t) / (Phi_gamma0 * T_REF)


def gamma_deph(t, Gdeph0, alpha_eff):
    return Gdeph0 * (1 + alpha_eff * fluence_fraction(t))


def qfi(N, t, Gdeph0, alpha_eff, tau=None):
    if tau is None:
        tau = 8.0e-3 / (N_ARRAY * 1.26e5)
    G = gamma_deph(t, Gdeph0, alpha_eff)
    return (N ** 2) * np.exp(-2 * N * G * tau)


# Nominal (calibrated / manuscript-stated) constant values
NOMINAL = dict(A_phi0=25e-6, kappa_eff=120.0, Gdeph0=1.26e5, alpha_eff=61.5)
PARAM_LABELS = {
    "A_phi0": r"$A_\varphi^{(0)}$ (fresh 1/f coeff., Table B.1)",
    "kappa_eff": r"$\kappa_{\rm eff}$ (calibrated damage coupling)",
    "Gdeph0": r"$\Gamma_{\rm deph}^{(0)}$ (fresh dephasing rate, manuscript)",
    "alpha_eff": r"$\alpha_{\rm eff}$ (calibrated dephasing-fluence coupling)",
}


def _qfi10(params: dict) -> float:
    return qfi(N_ARRAY, 10 * SECONDS_PER_YEAR, params["Gdeph0"], params["alpha_eff"])


def _fc10(params: dict) -> float:
    return corner_freq(10 * SECONDS_PER_YEAR, params["A_phi0"], params["kappa_eff"])


def _monte_carlo(n_draws: int = 1000, pct: float = 0.15, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_draws):
        params = {k: v * np.exp(rng.uniform(-np.log(1 + pct), np.log(1 + pct)))
                   for k, v in NOMINAL.items()}
        rows.append({**params, "QFI10": _qfi10(params), "fc10": _fc10(params)})
    return pd.DataFrame(rows)


def _tornado(pct: float = 0.15) -> pd.DataFrame:
    nominal_qfi = _qfi10(NOMINAL)
    rows = []
    for k in NOMINAL:
        lo_params = dict(NOMINAL); lo_params[k] = NOMINAL[k] * (1 - pct)
        hi_params = dict(NOMINAL); hi_params[k] = NOMINAL[k] * (1 + pct)
        qfi_lo, qfi_hi = _qfi10(lo_params), _qfi10(hi_params)
        rows.append(dict(param=k, low=qfi_lo, high=qfi_hi, nominal=nominal_qfi,
                          swing=abs(qfi_hi - qfi_lo)))
    return pd.DataFrame(rows).sort_values("swing", ascending=True)


def _panel_a_mc_qfi(ax, mc: pd.DataFrame) -> None:
    ax.hist(mc.QFI10, bins=40, color=sl.OKABE_ITO["blue"], alpha=0.75, edgecolor="white", linewidth=0.3)
    nominal = _qfi10(NOMINAL)
    ax.axvline(nominal, color="black", lw=1.4, ls="--", label=f"Nominal = {nominal:.1f}")
    ci_lo, ci_hi = np.percentile(mc.QFI10, [2.5, 97.5])
    ax.axvspan(ci_lo, ci_hi, color=sl.OKABE_ITO["blue"], alpha=0.12, label=f"95% CI [{ci_lo:.0f}, {ci_hi:.0f}]")
    ax.set_xlabel("QFI ($N{=}64$, 10 yr)")
    ax.set_ylabel("Monte Carlo draws")
    ax.legend(fontsize=6.2)


def _panel_b_mc_fc(ax, mc: pd.DataFrame) -> None:
    ax.hist(mc.fc10 / 1e6, bins=40, color=sl.OKABE_ITO["bluish_green"], alpha=0.75, edgecolor="white", linewidth=0.3)
    nominal = _fc10(NOMINAL) / 1e6
    ax.axvline(nominal, color="black", lw=1.4, ls="--", label=f"Nominal = {nominal:.2f} MHz")
    ci_lo, ci_hi = np.percentile(mc.fc10 / 1e6, [2.5, 97.5])
    ax.axvspan(ci_lo, ci_hi, color=sl.OKABE_ITO["bluish_green"], alpha=0.12,
               label=f"95% CI [{ci_lo:.1f}, {ci_hi:.1f}] MHz")
    ax.set_xlabel("Corner frequency (MHz, 10 yr)")
    ax.set_ylabel("Monte Carlo draws")
    ax.legend(fontsize=6.2)


def _panel_c_tornado(ax, tornado: pd.DataFrame) -> None:
    y = np.arange(len(tornado))
    nominal = tornado.nominal.iloc[0]
    for i, row in enumerate(tornado.itertuples()):
        ax.barh(i, row.high - nominal, left=nominal, color=sl.OKABE_ITO["vermillion"], alpha=0.8, height=0.55)
        ax.barh(i, row.low - nominal, left=nominal, color=sl.OKABE_ITO["blue"], alpha=0.8, height=0.55)
    ax.axvline(nominal, color="black", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([PARAM_LABELS[p] for p in tornado.param], fontsize=6.0)
    ax.set_xlabel("QFI ($N{=}64$, 10 yr)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=sl.OKABE_ITO["vermillion"], alpha=0.8, label="+15%"),
               plt.Rectangle((0, 0), 1, 1, color=sl.OKABE_ITO["blue"], alpha=0.8, label="-15%")]
    ax.legend(handles=handles, fontsize=6.2, loc="lower right")


def _panel_d_fan_chart(ax, n_boot: int = 300, seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    years = np.linspace(0, 10, 60)
    t_sec = years * SECONDS_PER_YEAR
    trajectories = np.zeros((n_boot, len(years)))
    for b in range(n_boot):
        params = {k: v * np.exp(rng.uniform(-np.log(1.15), np.log(1.15))) for k, v in NOMINAL.items()}
        trajectories[b, :] = [qfi(N_ARRAY, t, params["Gdeph0"], params["alpha_eff"]) for t in t_sec]
    nominal_traj = np.array([qfi(N_ARRAY, t, NOMINAL["Gdeph0"], NOMINAL["alpha_eff"]) for t in t_sec])
    for pct_lo, pct_hi, alpha in [(2.5, 97.5, 0.12), (16, 84, 0.22)]:
        lo = np.percentile(trajectories, pct_lo, axis=0)
        hi = np.percentile(trajectories, pct_hi, axis=0)
        ax.fill_between(years, lo, hi, color=sl.OKABE_ITO["reddish_purple"], alpha=alpha, linewidth=0)
    ax.plot(years, nominal_traj, color=sl.OKABE_ITO["reddish_purple"], lw=1.8, label="Nominal (calibrated)")
    ax.set_yscale("log")
    ax.set_xlabel("Operating time (years)")
    ax.set_ylabel("QFI ($N{=}64$)")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.text(0.02, 0.03, "Shaded: 68%/95% Monte Carlo bootstrap bands\n"
            "(+/-15% log-uniform perturbation of all 4 constants)",
            transform=ax.transAxes, fontsize=5.6, va="bottom")


def build_figure(bundle=None) -> str:
    sl.print_section_header("FIGURE 8: Monte Carlo Uncertainty Propagation & Parameter Sensitivity")
    mc = _monte_carlo()
    tornado = _tornado()
    sl.print_numeric_summary(mc, columns=["QFI10", "fc10"], label="Monte Carlo draws (n=1000, +/-15% perturbation)")
    print("\n  Tornado sensitivity (QFI @ 10yr, +/-15% one-at-a-time):")
    print(tornado[["param", "low", "nominal", "high", "swing"]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    fig = plt.figure(figsize=sl.figsize_grid(2, 2, panel_w=2.9, panel_h=2.25))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.40, hspace=0.55,
                            left=0.10, right=0.96, top=0.90, bottom=0.11)

    ax_a = fig.add_subplot(gs[0, 0]); _panel_a_mc_qfi(ax_a, mc); sl.label_panel(ax_a, "a")
    ax_b = fig.add_subplot(gs[0, 1]); _panel_b_mc_fc(ax_b, mc); sl.label_panel(ax_b, "b")
    ax_c = fig.add_subplot(gs[1, 0]); _panel_c_tornado(ax_c, tornado); sl.label_panel(ax_c, "c")
    ax_d = fig.add_subplot(gs[1, 1]); _panel_d_fan_chart(ax_d); sl.label_panel(ax_d, "d")

    fig.suptitle("Figure 8. Monte Carlo uncertainty propagation and one-at-a-time parameter sensitivity",
                  x=0.10, ha="left", fontsize=10.0, fontweight="bold", y=0.995)

    path = sl.save_publication_figure(
        fig, "Figure_08_sensitivity_uncertainty", "Monte Carlo uncertainty propagation and sensitivity",
        n_panels=4, n_data_points=len(mc) + len(tornado),
        notes="Live Monte Carlo (n=1000) + one-at-a-time tornado sensitivity over the "
              "physics model's 4 calibration/material constants (README.md Section 3).")
    plt.close(fig)
    return path


if __name__ == "__main__":
    build_figure(None)
