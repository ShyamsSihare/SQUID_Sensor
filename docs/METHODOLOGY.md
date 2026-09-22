# Methodology

Full derivation of every equation implemented in `src/squid_sim/`, the
constants used, and the three calibration choices made where the
manuscript's own equations could not be evaluated self-consistently as
literally typeset. This document is the single source of truth that
`README.md`, the notebook's markdown cells, and `latex/supplementary_figures_and_tables.tex`
all summarize.

---

## 1. SQUID flux-noise spectrum

### 1.1 Governing equation

$$
S_\Phi^{\mathrm{tot}}(f,t) \;=\; S_\Phi^{\mathrm{zpf}}(f) \;+\; S_\Phi^{\mathrm{th}}(f) \;+\; \frac{A_\varphi(t)}{f}
$$

**Thermal (Tesche–Clarke) term**, from the fluctuation–dissipation theorem
applied to the SQUID's effective $R$–$L$ shunt impedance:

$$
S_\Phi^{\mathrm{th}}(f) = \frac{16\,k_B T\,L_{\mathrm{sq}}^2/R_n}{1+\left(2\pi f L_{\mathrm{sq}}/R_n\right)^2}
$$

Implemented in `physics.py::s_thermal`, evaluated in SI (Wb²/Hz) then
converted to the $\Phi_0^2/\mathrm{Hz}$ convention used throughout
(dividing by $\Phi_0^2$, the flux-quantum squared) so it is directly
comparable to the manuscript's own $A_\varphi^{(0)}=25\,\mu\Phi_0^2/\mathrm{Hz}$
table value.

**Zero-point-fluctuation term.** The manuscript's literal
$S_\Phi^{\mathrm{zpf}}=\hbar\,\mathrm{Re}[Z_{\mathrm{eff}}]$ is not
dimensionally a flux-noise spectral density as typeset (ħ × Ω ≠ Wb²/Hz).
What the manuscript states unambiguously in prose, however, is that this
term is negligible at $T=4.2\,\mathrm{K}$ over the $0.01$–$10^4\,\mathrm{Hz}$
band (since $\hbar\omega \ll k_BT$ there — $k_BT/h\approx88\,\mathrm{GHz}$ at
4.2 K, far above this band). `physics.py::s_zpf` represents it as a fixed,
deliberately tiny fraction ($10^{-6}\times$) of the thermal floor, which
reproduces exactly that stated physical fact.

**1/f term**, from two-level-system (TLS) defects, growing with radiation
damage — see §2.

### 1.2 Corner frequency

$$
f_c(t) = A_\varphi(t) \,\big/\, S_\Phi^{\mathrm{th}}(f\!\to\!0)
$$

evaluated self-consistently in the $\Phi_0^2/\mathrm{Hz}$ unit convention
(`physics.py::corner_frequency`) rather than the manuscript's literal SI
formula, for the same unit-consistency reason as above.

---

## 2. Radiation-damage kinetics

### 2.1 Displacement damage dose (dpa)

The manuscript's `eq:ddd` integrates $\Phi_n(E,t)\,\sigma_{\mathrm{dpa}}(E)$
over energy and time and then divides by the atomic number density $n_{\mathrm{at}}$.
This division makes the result dimensionally NOT a dpa (displacements-per-atom,
dimensionless) quantity: $\Phi_n$ [cm⁻²s⁻¹] × $\sigma_{\mathrm{dpa}}$ [cm²]
already gives a dimensionless-per-second (dpa/s) rate; dividing by $n_{\mathrm{at}}$
[cm⁻³] introduces a spurious cm³ factor.

**Resolution used** (`physics.py::ddd`): the standard NRT dpa-rate
definition, with the manuscript's own arc-dpa recombination factor $\xi=0.30$
applied as stated:

$$
\mathcal{D}(t) = \xi\,\Phi_n\,\sigma_{\mathrm{dpa}}\,t, \qquad
\xi=0.30,\ \ \sigma_{\mathrm{dpa}}=600\,\mathrm{mb},\ \ \Phi_n = 1.2\times10^{11}\,\mathrm{cm^{-2}s^{-1}}
$$

### 2.2 1/f noise-amplitude growth (calibrated)

The manuscript's `eq:tls_noise` states
$A_\varphi(t)=A_\varphi^{(0)}\sqrt{1+\kappa_{\mathrm{TLS}}\,\mathcal{D}(t)}$,
with $\kappa_{\mathrm{TLS}}=5\times10^{20}\,\mathrm{cm^{-2}}$. Multiplying a
dimensionless $\mathcal{D}(t)$ (dpa) by a $\mathrm{cm^{-2}}$ constant does
**not** give a dimensionless bracket for any consistent reading of
$\mathcal{D}(t)$ — this is the second unit inconsistency found.

**Resolution used**: the functional *form* ($\sqrt{1+\kappa\,(\cdot)}$
growth) is kept exactly, but the coupling constant is replaced with a
dimensionless $\kappa_{\mathrm{eff}}$, calibrated so the ratio
$\mathcal{D}(t)/\mathcal{D}(T_{\mathrm{ref}})$ (normalized to the
manuscript's own stated $T_{\mathrm{ref}}=7\,\mathrm{yr}$ "pre-failure"
horizon, Sec. 6.2) reproduces the ≈11× 1/f-amplitude growth the
manuscript's *own* Fig. 6.1 plots between its "fresh" and "pre-failure"
curves:

$$
A_\varphi(t) = A_\varphi^{(0)}\sqrt{1+\kappa_{\mathrm{eff}}\,d(t)}, \qquad
d(t)\equiv \mathcal{D}(t)/\mathcal{D}(T_{\mathrm{ref}}), \qquad
\kappa_{\mathrm{eff}}=120 \ \ (\Rightarrow \sqrt{1+120}\approx 11)
$$

### 2.3 Critical-current degradation

Manuscript `eq:critical_current_degradation` uses coefficients
($\alpha_I=1.2\times10^{-20}\,\mathrm{cm^2}$, etc.) calibrated for a raw-DDD
scale the manuscript's own (corrected) `eq:ddd` never reaches within any
realistic operating lifetime — applied literally, they predict **zero**
measurable $I_c$ degradation. The same functional shape (polynomial +
log-saturation) is retained, evaluated against the normalized $d(t)$ scale
above, with order-1 coefficients chosen so $I_c$ degrades by a physically
typical few percent by $T_{\mathrm{ref}}$ (consistent with Nb-superconductor
radiation-damage literature):

$$
\frac{I_c(t)}{I_c(0)} = 1-\alpha_{\mathrm{eff}}\,d(t) - \gamma_I\ln\big(1+\delta_{\mathrm{eff}}\,d(t)\big),
\qquad \alpha_{\mathrm{eff}}=0.02,\ \delta_{\mathrm{eff}}=1.0,\ \gamma_I=0.08\ (\text{manuscript's own value, kept as-is})
$$

---

## 3. Entanglement-enhanced sensing (GHZ / QFI)

### 3.1 Exact closed form

For an $N$-qubit GHZ probe under independent single-qubit dephasing at rate
$\Gamma_{\mathrm{deph}}$, interrogated for time $\tau$, the manuscript's
Theorem `exact_qfi_dephased` gives the **exact** (not perturbative) quantum
Fisher information:

$$
\mathcal{F}_Q(N,\tau) = N^2\, e^{-2N\,\Gamma_{\mathrm{deph}}\,\tau}
$$

implemented verbatim in `physics.py::qfi_ghz` — no calibration needed here,
since this equation is dimensionally and physically self-consistent as
typeset.

### 3.2 Dephasing rate vs. accumulated dose (calibrated)

The manuscript states
$\Gamma_{\mathrm{deph}}(t) = \Gamma_{\mathrm{deph}}^{(0)} + \alpha_\varphi\,\mathcal{F}(t)$
with $\alpha_\varphi=1.0\times10^{-9}\,\mathrm{cm^2\,s^{-1}}$ and raw gamma
fluence $\mathcal{F}(t)=\Phi_\gamma t$. Using these literal values, the
dephasing rate exceeds the fresh rate within **minutes** of operation at the
manuscript's own $\Phi_\gamma=4.5\times10^{12}\,\mathrm{cm^{-2}s^{-1}}$ —
directly contradicting the manuscript's own 5–7 year operational narrative
(Sec. 6). This is the third inconsistency found.

**Resolution used**: same functional form (linear growth with fluence),
same normalization strategy as §2.2, calibrated so the fresh-sensor
operating point $N\Gamma_{\mathrm{deph}}^{(0)}\tau_{\mathrm{int}}=8\times10^{-3}$
(the manuscript's own quoted value, main text line ≈4126) is reproduced
exactly, and the dephasing rate at $T_{\mathrm{ref}}$ grows to give a
"graceful", multi-year degradation timescale consistent with the paper's
qualitative narrative:

$$
\Gamma_{\mathrm{deph}}(t) = \Gamma_{\mathrm{deph}}^{(0)}\big(1+\alpha_{\mathrm{eff}}\,\mathcal{F}(t)/\mathcal{F}(T_{\mathrm{ref}})\big),
\qquad \alpha_{\mathrm{eff}}=61.5,\ \ \Gamma_{\mathrm{deph}}^{(0)}/2\pi=20\,\mathrm{kHz}\ (\text{manuscript's own value})
$$

### 3.3 Entanglement visibility

$$
\mathcal{V}(\tau) = e^{-N\Gamma_{\mathrm{deph}}(t)\tau} = \sqrt{\mathcal{F}_Q(N,\tau)}\big/N
$$

The right-hand identity is an exact algebraic consequence of §3.1 and is
independently checked in `module_02_validation_suite.py::check_visibility_identity`
(max. discrepancy $<10^{-9}$ across all 63 $(N,t)$ grid points).

---

## 4. Quantum anomaly detection

### 4.1 Amplitude encoding

A spectrum $S_\Phi(f)$, sampled at 256 log-spaced frequency bins, is
amplitude-encoded into an $n=8$-qubit ($2^8=256$-dimensional) statevector.

**Naive per-sample-normalized encoding** — dividing each spectrum's
$\sqrt{S}$ vector by its own norm — was tried first, matching the
manuscript's Fig. `quantum_encoding` description literally. This provably
discards all information about the spectrum's *overall* power (only its
normalized shape survives). Numerically this was confirmed to be a real
problem: three of five failure modes (sensor degradation, radiation burst,
calibration drift) act mostly as a near-uniform rescaling of the whole
spectrum, not a shape change, because with this SQUID's own stated
parameters the 1/f term dominates the thermal floor across the *entire*
0.01–10⁴ Hz band — so a purely shape-based encoding measured AUC $=0.60$
on this dataset, versus $0.95$ for the classical (power-aware) baseline.

**Resolution used** (`detector.py::encode_amplitude`, `circuit.py::encode_batch`):
a magnitude-preserving encoding using 255 frequency bins + one fixed-reference
"magnitude ancilla" dimension (still exactly $2^8=256$), with a shared
(not per-sample) reference power scale
$\mathrm{ref\_total\_power}=4.0$, calibrated from a pooled sample spanning
healthy + all five failure modes (observed max total power $\approx2.52$,
comfortable headroom before any clipping). This restores AUC to $0.9486$,
parity with the classical baseline.

### 4.2 Quantum relative-entropy score

$$
\rho_{\mathrm{healthy}} = \frac{1}{M}\sum_{i=1}^{M}\ket{\psi_i}\!\bra{\psi_i} = \sum_k\lambda_k\ket{v_k}\!\bra{v_k}
$$

$$
\mathcal{A}(t) = D\big(\ket{\psi_t}\!\bra{\psi_t}\,\|\,\rho_{\mathrm{healthy}}\big) = -\sum_k\ln(\lambda_k)\,\big|\langle v_k|\psi_t\rangle\big|^2
$$

This is the manuscript's Umegaki relative-entropy anomaly score
(`eq:anomaly_score`), and per the manuscript's own optimality argument
(App. D.2) is what an optimally-trained VQC/SWAP-test detector converges
toward — used as the "quantum detector" for large-N benchmarking
(Figs. 2, 9), while literal small-scale VQC gradient-descent training
(§5) is reserved for the explicit hyperparameter-sweep experiment
(Fig. 5) where the training procedure itself is the object of study.

### 4.3 Classical baseline

An unsupervised One-Class SVM (RBF kernel), fit only on healthy-spectrum
classical features (total power, low/high-band power, spectral centroid,
peak-to-mean ratio, log-log slope) — symmetric in construction with the
quantum detector's unsupervised, healthy-only training.

---

## 5. Variational quantum circuit (VQC) training

### 5.1 Ansatz

A brick-wall circuit of single-qubit $R_Y,R_Z$ rotations per layer,
interleaved with a fixed brick-wall pattern of CZ entangling gates
(Sec. 4.3), simulated as a batched statevector (`circuit.py`), not a
dense $2^n\times2^n$ matrix, for tractable runtime up to $n=10$ qubits.

### 5.2 Loss and exact gradient

$$
\mathcal{L}(\boldsymbol\theta) = 1-\big|\langle\phi_{\mathrm{ref}}|U(\boldsymbol\theta)|\psi_{\mathrm{in}}\rangle\big|^2
$$

where $\ket{\phi_{\mathrm{ref}}}$ is the mean encoded direction of a
100-spectrum healthy reference subsample (found to give a well-conditioned,
rapidly-converging loss landscape; the literal fixed $\ket{0\cdots0}$ target
was tried first and found to give an almost-flat, uninformative loss
surface for this ansatz depth — documented in `circuit.py::train_vqc`).

$$
\frac{\partial\mathcal{L}}{\partial\theta_k} = \tfrac12\Big[\mathcal{L}(\theta_k+\tfrac\pi2)-\mathcal{L}(\theta_k-\tfrac\pi2)\Big]
$$

the manuscript's own exact parameter-shift rule (`eq:parameter_shift_rule`),
implemented exactly (not approximated by finite differences).

### 5.3 Reduced-scale training budget

| Manuscript | This repository (default) | Why |
|---|---|---|
| $M_{\mathrm{train}}=10{,}000$ | 3,000 (relative-entropy detector) / 200 (VQC sweep) | Eigendecomposition / per-config training cost; 3,000 already gives a converged 256×256 healthy density matrix |
| 10 training seeds | 3 seeds | Runtime (8 configs × 3 seeds × 60 parameter-shift iterations ≈ 10 min already) |
| — | 60 iterations, batch size 8 | Kept small enough to run interactively / in CI |

All of these are named constants at the top of `run_all.py` and
`notebooks/modules/module_fig05_vqc_hyperparameter_sweep.py` — a full-scale
rerun is a one-line change, given sufficient compute time.

---

## 6. Predictive maintenance (QPMS)

The three-channel composite index (Sec. 5.1):

$$
\mathrm{QPMS}(t) = w_A\,\frac{\mathcal{A}(t)}{\mathcal{A}_{\mathrm{thresh}}} + w_Q\Big(1-\frac{\mathcal{F}_Q(t)}{\mathcal{F}_Q(0)}\Big) + w_D\,\mathcal{D}(t), \qquad w_A{=}0.45,\ w_Q{=}0.35,\ w_D{=}0.20
$$

combining the anomaly channel (§4.2), the QFI-degradation channel (§3),
and the fluence channel (§2.1) — plotted as individual trajectories, not a
stacked area, since the composite is a weighted *average*, not a sum
(a stacked-area rendering was tried first and found visually misleading for
exactly this reason; see `module_fig04_qpms_trajectory.py`).

---

## 7. Validation

Every equation above has a corresponding independent-recomputation check in
`notebooks/modules/module_02_validation_suite.py` (mirrored as pytest cases
in `tests/test_validation_suite.py`), run automatically in CI on every push.
As of the last full run: **8/8 checks pass**. See `README.md`'s
"Reproducibility & validation" section for the summary and
`latex/supplementary_figures_and_tables.tex`'s Table S9 for the full,
publication-formatted results table.
