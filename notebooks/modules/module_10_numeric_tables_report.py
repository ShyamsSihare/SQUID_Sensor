from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(__file__))
import module_00_style_library as sl  # noqa: E402


def _render_table_page(pdf: PdfPages, df: pd.DataFrame, title: str, caption: str,
                        col_widths: list | None = None, fontsize: float = 7.5) -> None:
    n_rows = len(df) + 1
    fig_h = min(0.35 * n_rows + 1.2, 10.5)
    fig, ax = plt.subplots(figsize=(sl.COLUMN_WIDTH_DOUBLE_IN, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left", pad=14)

    cell_text = df.to_numpy()
    col_labels = list(df.columns)
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="upper center",
                    cellLoc="center", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.0, 1.35)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#2b2b2b")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f4f4f4" if row % 2 == 0 else "white")
    fig.text(0.02, 0.01, caption, fontsize=6.2, style="italic", color="#555555", wrap=True)
    sl.watermark_provenance(fig, f"Simulated data - {sl.MANUSCRIPT_TITLE} - {sl.GENERATED_ON}")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _fmt(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: f"{v:.{digits}g}")
    return out


def build_report(bundle, validation_result=None) -> str:
    sl.print_section_header("SUPPLEMENTARY NUMERIC TABLES REPORT (PDF)")
    out_path = os.path.join(sl.FIGDIR, "Supplementary_Numeric_Tables.pdf")

    with PdfPages(out_path) as pdf:
        _render_table_page(
            pdf, bundle.squid_params, "Table S1. SQUID simulation parameters",
            "Manuscript Table 3 (tab:simulation_params) constants, reproduced for reference.")

        _render_table_page(
            pdf, bundle.material_params, "Table S2. Material / radiation-damage parameters (Nb)",
            "Manuscript Table B.1 (tab:material_params) constants, reproduced for reference.",
            fontsize=7.0)

        g = bundle.noise_spectra.copy()
        g["sqrt_S_phi_uPhi0"] = g["sqrt_S_phi_Phi0_per_sqrtHz"] * 1e6
        s3 = g.groupby("stage")["sqrt_S_phi_uPhi0"].agg(["count", "mean", "std", "min", "max"]).reset_index()
        s3 = s3.rename(columns={"stage": "Lifecycle stage", "count": "N snapshots x freq bins"})
        _render_table_page(pdf, _fmt(s3), "Table S3. Noise-spectrum summary by lifecycle stage",
                            r"$\sqrt{S_\Phi}$ in $\mu\Phi_0/\sqrt{\rm Hz}$, aggregated over all 256 frequency "
                            "bins x 20 repeated snapshots per stage.")

        s4 = bundle.lifecycle.iloc[[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 199]][
            ["years", "ddd_dpa", "Ic_uA", "A_phi_Phi0^2_per_Hz", "corner_freq_Hz",
             "QFI_N64", "entanglement_visibility_N64"]
        ].reset_index(drop=True)
        s4 = s4.rename(columns={
            "years": "Years", "ddd_dpa": "DDD (dpa)", "Ic_uA": "Ic (uA)",
            "A_phi_Phi0^2_per_Hz": "A_phi", "corner_freq_Hz": "f_c (Hz)",
            "QFI_N64": "QFI (N=64)", "entanglement_visibility_N64": "Visibility",
        })
        _render_table_page(pdf, _fmt(s4, digits=5), "Table S4. Sensor lifecycle-trace milestones",
                            "11 evenly-spaced samples from the full 200-point, 0-10 year continuous trace. "
                            "A_phi in Phi0^2/Hz.", fontsize=7.0)

        s5 = bundle.qfi_vs_N_dose[bundle.qfi_vs_N_dose.years == 0][["N", "QFI", "visibility"]].reset_index(drop=True)
        _render_table_page(pdf, _fmt(s5, digits=6), "Table S5. QFI vs. array size N (fresh sensor, t=0)",
                            "Exact closed form QFI=N^2 exp(-2N*Gamma_deph(0)*tau_int) (eq:exact_qfi_dephased).")

        s6 = bundle.perf_by_mode.copy()
        s6["failure_mode"] = s6["failure_mode"].map(sl.FAILURE_MODE_LABELS)
        _render_table_page(pdf, _fmt(s6, digits=4), "Table S6. Anomaly-detection performance by failure mode",
                            "TPR/AUC/FPR at the fixed 99.9th-percentile operating threshold; "
                            "22,500 labelled test spectra.")

        s7 = bundle.early_warning.copy()
        s7["failure_mode"] = s7["failure_mode"].map(sl.FAILURE_MODE_LABELS)
        _render_table_page(pdf, _fmt(s7, digits=4), "Table S7. Early-warning horizon by failure mode",
                            "Simulated onset trajectories, hours before a fixed hard classical (3x mean-power) "
                            "alarm trip point.")

        _render_table_page(pdf, _fmt(bundle.vqc_sweep, digits=5),
                            "Table S8. VQC hyperparameter sweep (full results)",
                            "Real parameter-shift-rule gradient-descent training; 8 configs x 3 seeds "
                            "(reduced from manuscript's 10 seeds; see README.md Section 5).")

        if validation_result is not None:
            vdf = pd.DataFrame([{"Check": name, "Result": "PASS" if p else "FAIL", "Detail": d}
                                 for name, p, d in validation_result.checks])
            _render_table_page(pdf, vdf, "Table S9. Numerical-accuracy validation-suite results",
                                "Independent first-principles recomputation checks (module_02_validation_suite.py).",
                                fontsize=6.6)

    print(f"  Supplementary numeric-tables PDF written: {out_path} (9 pages)")
    return out_path


if __name__ == "__main__":
    import module_01_data_loader as dl
    import module_02_validation_suite as vs
    bundle = dl.load_all_datasets()
    vr = vs.run_validation_suite(bundle)
    build_report(bundle, vr)
