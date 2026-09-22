# Data Dictionary

Column-by-column description of every table in `data/`. All tables are
outputs of `src/squid_sim/run_all.py` (master seed `20260921`); see
`docs/METHODOLOGY.md` for the equations each column implements.

---

## `00_squid_simulation_parameters.csv` (13 rows)
Input constants (manuscript Table 3), not simulation output.

| Column | Meaning |
|---|---|
| `parameter` | Human-readable name |
| `symbol` | Mathematical symbol used in `docs/METHODOLOGY.md` |
| `value` | Numeric value |
| `unit` | Physical unit |

---

## `00_material_damage_parameters.csv` (8 rows)
Niobium material constants (manuscript Table B.1), input constants.

| Column | Meaning |
|---|---|
| `parameter` | Human-readable name |
| `value` | Numeric value |
| `unit` | Physical unit |

---

## `01_squid_noise_spectra.csv` (20,480 rows = 4 stages × 20 reps × 256 freq bins)
Simulated flux-noise spectra at four lifecycle stages, from `physics.py::total_noise_spectrum` + `spectra.py::_jitter`.

| Column | Meaning |
|---|---|
| `stage` | One of `fresh`, `2yr`, `5yr_precursor`, `7yr_prefail` |
| `years` | Operating time in years for that stage (0, 2, 5, 7) |
| `rep` | Repeated-snapshot index (0–19); models measurement-to-measurement variability at fixed `t` |
| `freq_hz` | Frequency bin (256 log-spaced points, 0.01–10,000 Hz) |
| `S_phi_Phi0^2_per_Hz` | Flux-noise power spectral density $S_\Phi(f)$, in $\Phi_0^2/\mathrm{Hz}$ |
| `sqrt_S_phi_Phi0_per_sqrtHz` | $\sqrt{S_\Phi(f)}$, the amplitude spectral density, in $\Phi_0/\sqrt{\mathrm{Hz}}$ (multiply by $10^6$ for $\mu\Phi_0/\sqrt{\mathrm{Hz}}$, the unit used in all figures/tables) |

---

## `02_sensor_lifecycle_trace.csv` (200 rows)
Continuous trace over `years` $\in[0,10]$ (200 evenly-spaced points).

| Column | Meaning | Equation |
|---|---|---|
| `years` | Operating time | — |
| `ddd_dpa` | Displacement damage dose $\mathcal{D}(t)$, in dpa | §2.1 |
| `damage_fraction` | $\mathcal{D}(t)/\mathcal{D}(T_{\mathrm{ref}})$, dimensionless, $T_{\mathrm{ref}}=7\,\mathrm{yr}$ | §2.1 |
| `Ic_uA` | Critical current $I_c(t)$, µA | §2.3 |
| `A_phi_Phi0^2_per_Hz` | 1/f noise-amplitude coefficient $A_\varphi(t)$ | §2.2 |
| `corner_freq_Hz` | Corner frequency $f_c(t)$, Hz | §1.2 |
| `Gamma_deph_per_s` | Dephasing rate $\Gamma_{\mathrm{deph}}(t)$, s⁻¹ | §3.2 |
| `QFI_N64` | Quantum Fisher information at $N=64$ | §3.1 |
| `entanglement_visibility_N64` | $\mathcal{V}(\tau)$ at $N=64$ | §3.3 |

---

## `03_qfi_vs_N_and_dose.csv` (63 rows = 9 values of N × 7 years)
| Column | Meaning |
|---|---|
| `N` | GHZ array size (1, 2, 4, 8, 16, 32, 64, 128, 256) |
| `years` | Operating time snapshot (0, 1, 2, 3, 5, 7, 10) |
| `QFI` | $\mathcal{F}_Q(N,\tau)$, exact closed form (§3.1) |
| `visibility` | $\mathcal{V}(\tau)=\sqrt{\mathcal{F}_Q}/N$ (§3.3) |
| `tau_int_s` | Interrogation time used, seconds (fixed, calibrated so the fresh $N{=}64$ operating point matches the manuscript's stated $N\Gamma_{\mathrm{deph}}^{(0)}\tau=8\times10^{-3}$) |

---

## `04_anomaly_scores_full_testset.csv` (22,500 rows = 20,000 healthy + 5×500 anomalous)
Per-sample detector outputs on the labelled test set.

| Column | Meaning |
|---|---|
| `sample_id` | Row index |
| `true_label` | 0 = healthy, 1 = anomalous |
| `failure_mode` | `healthy`, `sensor_degradation`, `radiation_burst`, `coolant_anomaly`, `foreign_material`, or `calibration_drift` |
| `quantum_relentropy_score` | $\mathcal{A}(t)$, Umegaki relative entropy (§4.2) — higher = more anomalous |
| `classical_ocsvm_score` | Classical One-Class-SVM decision score (§4.3), sign-flipped so higher = more anomalous |

---

## `05_roc_curve_overall.csv` (45,004 rows)
Full ROC curves for both detectors, pooled across all failure modes.

| Column | Meaning |
|---|---|
| `detector` | `quantum` or `classical` |
| `fpr` | False-positive rate at this threshold |
| `tpr` | True-positive rate at this threshold |

AUC = `numpy.trapezoid(tpr, fpr)` after sorting by `fpr` (see any figure module for the exact call).

---

## `06_anomaly_performance_by_mode.csv` (5 rows, one per failure mode)
| Column | Meaning |
|---|---|
| `failure_mode` | One of the 5 failure modes |
| `TPR_quantum_pct` / `TPR_classical_pct` | True-positive rate (%) at the fixed 99.9th-percentile operating threshold |
| `AUC_quantum` / `AUC_classical` | Per-mode AUC (this mode vs. healthy only) |
| `FPR_quantum_x1e-4` / `FPR_classical_x1e-4` | Overall false-positive rate, in units of ×10⁻⁴ |

---

## `07_early_warning_horizon.csv` (5 rows)
Simulated onset trajectories (25 per mode): linear ramp from healthy to full
failure-mode severity over 120 simulated hours, scored continuously by both
detectors and a fixed "hard" classical alarm (total power > 3× healthy mean).

| Column | Meaning |
|---|---|
| `failure_mode` | One of the 5 failure modes |
| `EW_quantum_hours_mean` / `_std` | Mean/s.d. hours the quantum detector's threshold-crossing precedes the hard alarm |
| `n_detected_quantum` | Number of the 25 trajectories where the quantum detector fired before the hard alarm |
| `EW_classical_hours_mean` / `_std` / `n_detected_classical` | Same, for the classical statistical detector |

---

## `08_qpms_trajectory.csv` (400 rows)
Continuous 5-year QPMS simulation (§6), 400 evenly-spaced days.

| Column | Meaning |
|---|---|
| `day` / `year` | Time since commissioning |
| `anomaly_channel` | $\mathcal{A}(t)/\mathcal{A}_{\mathrm{thresh}}$, capped at 1.2 |
| `qfi_degradation_channel` | $1-\mathcal{F}_Q(t)/\mathcal{F}_Q(0)$ |
| `fluence_channel` | $\mathcal{D}(t)$, capped at 1.0 |
| `qpms_scalar` | The weighted composite $\mathrm{QPMS}(t)$ |

---

## `09_early_detection_probability_vs_N.csv` (75 rows = 3 values of N × 25 horizons)
| Column | Meaning |
|---|---|
| `N` | Array size (4, 16, or 64) |
| `hours_before_hard_alarm` | Warning horizon, hours (0–96) |
| `detection_probability` | Modeled sigmoid detection probability at that horizon, scaled by $\sqrt{N/64}$ per the GHZ $N^2$ QFI-scaling argument |

---

## `10_vqc_hyperparameter_sweep.csv` (8 rows, one per hyperparameter configuration)
Real gradient-descent training results (§5), 3 seeds per row.

| Column | Meaning |
|---|---|
| `L` | Circuit depth (number of ansatz layers) |
| `n` | Number of qubits |
| `lr` | Learning rate $\eta$ |
| `AUC_mean` / `AUC_std` | Validation AUC, mean/s.d. over 3 seeds |
| `FAR_at_95pct_thresh_mean` / `_std` | False-alarm rate at the 95th-percentile validation threshold |
| `n_seeds` | Number of training seeds averaged (3) |

---

## Reading these tables in one line

```python
import pandas as pd
df = pd.read_csv("data/06_anomaly_performance_by_mode.csv")
```

or, for the full validated bundle with structural checks:

```python
import sys; sys.path.insert(0, "notebooks/modules")
import module_01_data_loader as dl
bundle = dl.load_all_datasets(data_dir="data")   # raises if any table fails validation
```
