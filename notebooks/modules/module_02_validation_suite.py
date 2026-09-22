from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import module_00_style_library as sl  # noqa: E402

TOL_REL = 1e-6


class ValidationResult:
    def __init__(self):
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append((name, passed, detail))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))

    def summary(self) -> tuple[int, int]:
        n_pass = sum(1 for _, p, _ in self.checks if p)
        n_total = len(self.checks)
        return n_pass, n_total


def _rel_err(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom


def check_qfi_closed_form(bundle, vr: ValidationResult) -> None:
    lifecycle = bundle.lifecycle
    tau_int = 8.0e-3 / (64 * 1.26e5)
    sample_idx = [0, 50, 100, 150, 199]
    max_err = 0.0
    for idx in sample_idx:
        row = lifecycle.iloc[idx]
        G = row["Gamma_deph_per_s"]
        qfi_recomputed = 64 ** 2 * np.exp(-2 * 64 * G * tau_int)
        qfi_saved = row["QFI_N64"]
        err = _rel_err(qfi_recomputed, qfi_saved)
        max_err = max(max_err, err)
    vr.check("QFI closed-form recomputation (N=64, 5 spot checks)",
              max_err < 1e-3, f"max relative error = {max_err:.2e}")


def check_visibility_identity(bundle, vr: ValidationResult) -> None:
    qfi_df = bundle.qfi_vs_N_dose
    recomputed_v = np.sqrt(qfi_df.QFI.to_numpy()) / qfi_df.N.to_numpy()
    max_err = np.max(np.abs(recomputed_v - qfi_df.visibility.to_numpy()))
    vr.check("Entanglement-visibility identity V=sqrt(QFI)/N (all N, all years)",
              max_err < 1e-9, f"max abs error = {max_err:.2e}")


def check_roc_monotonicity(bundle, vr: ValidationResult) -> None:
    roc = bundle.roc_curves
    all_monotonic = True
    detail = ""
    for det in roc.detector.unique():
        sub = roc[roc.detector == det].sort_values("fpr")
        tpr = sub.tpr.to_numpy()
        diffs = np.diff(tpr)
        n_violations = int((diffs < -1e-12).sum())
        if n_violations > 0:
            all_monotonic = False
            detail += f"{det}: {n_violations} TPR decreases; "
    vr.check("ROC curve TPR monotonicity (non-decreasing with FPR, both detectors)",
              all_monotonic, detail or "no violations")


def check_auc_recomputation(bundle, vr: ValidationResult) -> None:
    roc = bundle.roc_curves
    ok = True
    detail_parts = []
    for det in roc.detector.unique():
        sub = roc[roc.detector == det].sort_values("fpr")
        auc = np.trapezoid(sub.tpr, sub.fpr)
        in_range = 0.0 <= auc <= 1.0
        ok = ok and in_range
        detail_parts.append(f"{det} AUC={auc:.4f}")
    vr.check("Recomputed overall AUC lies in valid [0,1] range (both detectors)",
              ok, "; ".join(detail_parts))


def check_corner_frequency_identity(bundle, vr: ValidationResult) -> None:
    lifecycle = bundle.lifecycle
    idx1, idx2 = 20, 180
    floor1 = lifecycle.iloc[idx1]["A_phi_Phi0^2_per_Hz"] / lifecycle.iloc[idx1]["corner_freq_Hz"]
    floor2 = lifecycle.iloc[idx2]["A_phi_Phi0^2_per_Hz"] / lifecycle.iloc[idx2]["corner_freq_Hz"]
    err = _rel_err(floor1, floor2)
    vr.check("Corner-frequency identity f_c=A_phi/floor implies constant floor "
              "(2 independent time points)", err < 1e-6, f"relative discrepancy = {err:.2e}")


def check_probability_bounds(bundle, vr: ValidationResult) -> None:
    checks = [
        ("entanglement_visibility_N64 in [0,1]", bundle.lifecycle.entanglement_visibility_N64, 0, 1),
        ("visibility in [0,1]", bundle.qfi_vs_N_dose.visibility, 0, 1),
        ("TPR_quantum_pct in [0,100]", bundle.perf_by_mode.TPR_quantum_pct, 0, 100),
        ("TPR_classical_pct in [0,100]", bundle.perf_by_mode.TPR_classical_pct, 0, 100),
        ("AUC_quantum in [0,1]", bundle.perf_by_mode.AUC_quantum, 0, 1),
        ("AUC_classical in [0,1]", bundle.perf_by_mode.AUC_classical, 0, 1),
        ("detection_probability in [0,1]", bundle.detect_prob_vs_N.detection_probability, 0, 1),
        ("qpms_scalar >= 0", bundle.qpms_trajectory.qpms_scalar, 0, np.inf),
        ("AUC_mean (VQC sweep) in [0,1]", bundle.vqc_sweep.AUC_mean, 0, 1),
    ]
    all_ok = True
    for name, series, lo, hi in checks:
        ok = bool(((series >= lo - 1e-9) & (series <= hi + 1e-9)).all())
        all_ok = all_ok and ok
        if not ok:
            vr.check(name, False, f"min={series.min():.4g}, max={series.max():.4g}, expected [{lo},{hi}]")
    vr.check("All probability/fraction/rate columns within documented valid ranges (9 checks)",
              all_ok)


def check_cross_table_consistency(bundle, vr: ValidationResult) -> None:
    scores = bundle.anomaly_scores
    n_healthy = (scores.true_label == 0).sum()
    n_anomaly = (scores.true_label == 1).sum()
    n_modes_observed = scores[scores.true_label == 1].failure_mode.nunique()
    ok = (n_modes_observed == 5) and (n_healthy > 0) and (n_anomaly > 0)
    vr.check("Anomaly-scores table has 5 distinct failure modes + healthy class",
              ok, f"healthy n={n_healthy}, anomaly n={n_anomaly}, modes={n_modes_observed}")

    perf_modes = set(bundle.perf_by_mode.failure_mode)
    score_modes = set(scores[scores.true_label == 1].failure_mode.unique())
    ok2 = perf_modes == score_modes
    vr.check("Failure-mode set matches between anomaly_scores and perf_by_mode tables",
              ok2, f"perf_by_mode={sorted(perf_modes)}, anomaly_scores={sorted(score_modes)}")


def run_validation_suite(bundle) -> ValidationResult:
    sl.print_section_header("NUMERICAL-ACCURACY VALIDATION SUITE (independent recomputation checks)")
    vr = ValidationResult()
    check_qfi_closed_form(bundle, vr)
    check_visibility_identity(bundle, vr)
    check_roc_monotonicity(bundle, vr)
    check_auc_recomputation(bundle, vr)
    check_corner_frequency_identity(bundle, vr)
    check_probability_bounds(bundle, vr)
    check_cross_table_consistency(bundle, vr)
    n_pass, n_total = vr.summary()
    print(f"\n  VALIDATION SUMMARY: {n_pass}/{n_total} checks passed.")
    if n_pass < n_total:
        print("  ** Some checks failed -- review before using this dataset in the manuscript. **")
    else:
        print("  All independent-recomputation checks passed.")
    return vr


if __name__ == "__main__":
    import module_01_data_loader as dl
    bundle = dl.load_all_datasets()
    run_validation_suite(bundle)
