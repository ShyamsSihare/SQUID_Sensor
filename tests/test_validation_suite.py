import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "notebooks", "modules"))
DATA_DIR = os.path.join(REPO_ROOT, "data")

import module_01_data_loader as dl  # noqa: E402
import module_02_validation_suite as vs  # noqa: E402


@pytest.fixture(scope="module")
def bundle():
    return dl.load_all_datasets(data_dir=DATA_DIR, verbose=False)


@pytest.fixture(scope="module")
def validation_result(bundle):
    return vs.run_validation_suite(bundle)


# ------------------------------------------------------------------------
# Structural sanity tests (fast, don't need the full validation suite)
# ------------------------------------------------------------------------

def test_all_expected_files_present():
    for key, fname in dl.FILES.items():
        assert os.path.exists(os.path.join(DATA_DIR, fname)), f"missing {fname}"


def test_no_missing_columns(bundle):
    for name in dl.FILES:
        df = getattr(bundle, name)
        problems = dl._validate(name, df)
        assert not problems, f"{name}: {problems}"


def test_total_row_count_matches_expected(bundle):
    # Loose lower bound -- guards against a truncated/corrupted regeneration,
    # not an exact pin (exact counts documented in docs/DATA_DICTIONARY.md).
    assert bundle.total_rows() > 80_000


def test_five_failure_modes_present(bundle):
    modes = set(bundle.anomaly_scores[bundle.anomaly_scores.true_label == 1].failure_mode.unique())
    assert modes == {
        "sensor_degradation", "radiation_burst", "coolant_anomaly",
        "foreign_material", "calibration_drift",
    }


# ------------------------------------------------------------------------
# Independent first-principles accuracy checks (the real validation suite)
# ------------------------------------------------------------------------

def test_qfi_closed_form(bundle):
    tau_int = 8.0e-3 / (64 * 1.26e5)
    lifecycle = bundle.lifecycle
    for idx in [0, 50, 100, 150, 199]:
        row = lifecycle.iloc[idx]
        recomputed = 64 ** 2 * np.exp(-2 * 64 * row["Gamma_deph_per_s"] * tau_int)
        rel_err = abs(recomputed - row["QFI_N64"]) / max(abs(recomputed), abs(row["QFI_N64"]), 1e-300)
        assert rel_err < 1e-3, f"QFI mismatch at row {idx}: {rel_err:.2e}"


def test_visibility_identity(bundle):
    qfi_df = bundle.qfi_vs_N_dose
    recomputed_v = np.sqrt(qfi_df.QFI.to_numpy()) / qfi_df.N.to_numpy()
    max_err = np.max(np.abs(recomputed_v - qfi_df.visibility.to_numpy()))
    assert max_err < 1e-9


def test_roc_monotonicity(bundle):
    roc = bundle.roc_curves
    for det in roc.detector.unique():
        sub = roc[roc.detector == det].sort_values("fpr")
        assert (np.diff(sub.tpr.to_numpy()) >= -1e-12).all(), f"{det} ROC TPR decreased"


def test_auc_bounds(bundle):
    roc = bundle.roc_curves
    for det in roc.detector.unique():
        sub = roc[roc.detector == det].sort_values("fpr")
        auc = np.trapezoid(sub.tpr, sub.fpr)
        assert 0.0 <= auc <= 1.0


def test_probability_columns_in_range(bundle):
    checks = [
        (bundle.lifecycle.entanglement_visibility_N64, 0, 1),
        (bundle.qfi_vs_N_dose.visibility, 0, 1),
        (bundle.perf_by_mode.TPR_quantum_pct, 0, 100),
        (bundle.perf_by_mode.AUC_quantum, 0, 1),
        (bundle.detect_prob_vs_N.detection_probability, 0, 1),
        (bundle.vqc_sweep.AUC_mean, 0, 1),
    ]
    for series, lo, hi in checks:
        assert ((series >= lo - 1e-9) & (series <= hi + 1e-9)).all()


def test_full_validation_suite_all_pass(validation_result):
    n_pass, n_total = validation_result.summary()
    failed = [name for name, p, _ in validation_result.checks if not p]
    assert n_pass == n_total, f"{n_total - n_pass} check(s) failed: {failed}"
