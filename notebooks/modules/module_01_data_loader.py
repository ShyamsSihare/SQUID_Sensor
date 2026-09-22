from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import module_00_style_library as sl  # noqa: E402

try:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _THIS_DIR = os.getcwd()  # running inside a Jupyter cell -- no __file__
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
DATA_DIR = os.environ.get(
    "SQUID_DATA_DIR", os.path.join(_REPO_ROOT, "data")
)

DATA_PROVENANCE_NOTE = (
    "All values below are outputs of a numerical simulation that implements "
    "the manuscript's own physical equations (SQUID Tesche-Clarke thermal "
    "noise, TLS 1/f damage kinetics, exact GHZ dephased-QFI closed form, "
    "Umegaki relative-entropy quantum anomaly score, and parameter-shift "
    "VQC training) -- not hardware or field measurements. Three internal "
    "unit inconsistencies in the manuscript's own equations were found and "
    "transparently documented/calibrated during simulation; see README.md."
)

# ------------------------------------------------------------------------------
# 1. FILE MANIFEST
# ------------------------------------------------------------------------------

FILES: Dict[str, str] = {
    "squid_params":        "00_squid_simulation_parameters.csv",
    "material_params":     "00_material_damage_parameters.csv",
    "noise_spectra":       "01_squid_noise_spectra.csv",
    "lifecycle":           "02_sensor_lifecycle_trace.csv",
    "qfi_vs_N_dose":       "03_qfi_vs_N_and_dose.csv",
    "anomaly_scores":      "04_anomaly_scores_full_testset.csv",
    "roc_curves":          "05_roc_curve_overall.csv",
    "perf_by_mode":        "06_anomaly_performance_by_mode.csv",
    "early_warning":       "07_early_warning_horizon.csv",
    "qpms_trajectory":     "08_qpms_trajectory.csv",
    "detect_prob_vs_N":    "09_early_detection_probability_vs_N.csv",
    "vqc_sweep":           "10_vqc_hyperparameter_sweep.csv",
}

REQUIRED_COLUMNS: Dict[str, list] = {
    "squid_params":     ["parameter", "symbol", "value", "unit"],
    "material_params":  ["parameter", "value", "unit"],
    "noise_spectra":    ["stage", "years", "rep", "freq_hz",
                          "S_phi_Phi0^2_per_Hz", "sqrt_S_phi_Phi0_per_sqrtHz"],
    "lifecycle":        ["years", "ddd_dpa", "damage_fraction", "Ic_uA",
                          "A_phi_Phi0^2_per_Hz", "corner_freq_Hz",
                          "Gamma_deph_per_s", "QFI_N64", "entanglement_visibility_N64"],
    "qfi_vs_N_dose":    ["N", "years", "QFI", "visibility", "tau_int_s"],
    "anomaly_scores":   ["sample_id", "true_label", "failure_mode",
                          "quantum_relentropy_score", "classical_ocsvm_score"],
    "roc_curves":       ["detector", "fpr", "tpr"],
    "perf_by_mode":     ["failure_mode", "TPR_quantum_pct", "TPR_classical_pct",
                          "AUC_quantum", "AUC_classical"],
    "early_warning":    ["failure_mode", "EW_quantum_hours_mean", "EW_classical_hours_mean"],
    "qpms_trajectory":  ["day", "year", "anomaly_channel", "qfi_degradation_channel",
                          "fluence_channel", "qpms_scalar"],
    "detect_prob_vs_N": ["N", "hours_before_hard_alarm", "detection_probability"],
    "vqc_sweep":        ["L", "n", "lr", "AUC_mean", "AUC_std"],
}


@dataclass
class DatasetBundle:
    squid_params: pd.DataFrame
    material_params: pd.DataFrame
    noise_spectra: pd.DataFrame
    lifecycle: pd.DataFrame
    qfi_vs_N_dose: pd.DataFrame
    anomaly_scores: pd.DataFrame
    roc_curves: pd.DataFrame
    perf_by_mode: pd.DataFrame
    early_warning: pd.DataFrame
    qpms_trajectory: pd.DataFrame
    detect_prob_vs_N: pd.DataFrame
    vqc_sweep: pd.DataFrame

    def total_rows(self) -> int:
        total = 0
        for f in self.__dataclass_fields__:
            total += len(getattr(self, f))
        return total


# ------------------------------------------------------------------------------
# 2. VALIDATION
# ------------------------------------------------------------------------------

def _validate(name: str, df: pd.DataFrame) -> list:
    problems = []
    required = REQUIRED_COLUMNS.get(name, [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        problems.append(f"[{name}] missing required columns: {missing}")
    if df.empty:
        problems.append(f"[{name}] dataframe is EMPTY")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric_cols:
        n_nan = df[c].isna().sum()
        n_inf = np.isinf(df[c].to_numpy(dtype=float, na_value=0.0)).sum()
        if n_nan:
            problems.append(f"[{name}.{c}] {n_nan} NaN values")
        if n_inf:
            problems.append(f"[{name}.{c}] {n_inf} +/-inf values")
    return problems


def load_all_datasets(data_dir: str = DATA_DIR, verbose: bool = True) -> DatasetBundle:
    sl.print_section_header("DATA PROVENANCE & VALIDATION REPORT")
    print(DATA_PROVENANCE_NOTE)
    print(f"\nData directory: {data_dir}\n")

    loaded: Dict[str, pd.DataFrame] = {}
    all_problems: list = []
    for key, fname in FILES.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required dataset not found: {path}")
        df = pd.read_csv(path)
        loaded[key] = df
        problems = _validate(key, df)
        all_problems.extend(problems)
        status = "OK" if not problems else "ISSUES FOUND"
        if verbose:
            print(f"  [{status:12s}] {fname:45s} shape={df.shape}")

    if all_problems:
        print("\n  VALIDATION WARNINGS:")
        for p in all_problems:
            print(f"    - {p}")
    else:
        print("\n  All datasets passed structural validation (no missing "
              "columns, no NaN/inf in numeric fields).")

    bundle = DatasetBundle(**loaded)
    print(f"\n  Total rows across all {len(FILES)} tables: {bundle.total_rows():,}")
    return bundle


# ------------------------------------------------------------------------------
# 3. NUMERIC-VALUE REPORT (printed once, referenced by every figure)
# ------------------------------------------------------------------------------

def print_full_numeric_report(bundle: DatasetBundle) -> None:
    sl.print_section_header("FULL NUMERIC VALUE REPORT (all tables)")

    print("\n[1] SQUID simulation parameters (manuscript Table 3, tab:simulation_params):")
    print(bundle.squid_params.to_string(index=False))

    print("\n[2] Material / radiation-damage parameters (manuscript Table B.1, Nb):")
    print(bundle.material_params.to_string(index=False))

    print("\n[3] SQUID noise-spectrum dataset -- per-stage summary "
          "(sqrt(S_phi) in uPhi0/sqrt(Hz)):")
    g = bundle.noise_spectra.copy()
    g["sqrt_S_phi_uPhi0"] = g["sqrt_S_phi_Phi0_per_sqrtHz"] * 1e6
    summary = g.groupby("stage")["sqrt_S_phi_uPhi0"].agg(["count", "mean", "std", "min", "max"])
    print(summary.to_string(float_format=lambda v: f"{v:.4g}"))

    print("\n[4] Sensor lifecycle trace -- key milestones:")
    milestones = bundle.lifecycle.iloc[[0, 40, 100, 140, 199]][
        ["years", "ddd_dpa", "Ic_uA", "A_phi_Phi0^2_per_Hz",
         "corner_freq_Hz", "QFI_N64", "entanglement_visibility_N64"]
    ]
    print(milestones.to_string(index=False, float_format=lambda v: f"{v:.5g}"))

    print("\n[5] QFI vs N (fresh sensor, year=0):")
    qfi0 = bundle.qfi_vs_N_dose[bundle.qfi_vs_N_dose.years == 0][["N", "QFI", "visibility"]]
    print(qfi0.to_string(index=False, float_format=lambda v: f"{v:.5g}"))

    print("\n[6] Anomaly-detection score summary by class:")
    sc = bundle.anomaly_scores
    summary6 = sc.groupby("failure_mode")[
        ["quantum_relentropy_score", "classical_ocsvm_score"]
    ].agg(["count", "mean", "std"])
    print(summary6.to_string(float_format=lambda v: f"{v:.5g}"))

    print("\n[7] Per-failure-mode detection performance (operating threshold):")
    print(bundle.perf_by_mode.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print("\n[8] Early-warning horizon (hours before hard classical alarm):")
    print(bundle.early_warning.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print("\n[9] QPMS trajectory -- endpoints:")
    q_ends = bundle.qpms_trajectory.iloc[[0, -1]][
        ["year", "anomaly_channel", "qfi_degradation_channel", "fluence_channel", "qpms_scalar"]
    ]
    print(q_ends.to_string(index=False, float_format=lambda v: f"{v:.5g}"))

    print("\n[10] VQC hyperparameter sweep -- full table:")
    print(bundle.vqc_sweep.to_string(index=False, float_format=lambda v: f"{v:.5g}"))

    print("\n" + "=" * 88 + "\nEND OF FULL NUMERIC VALUE REPORT\n" + "=" * 88)


if __name__ == "__main__":
    bundle = load_all_datasets()
    print_full_numeric_report(bundle)
