
import os
import nbformat as nbf

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(PROJECT_DIR, "modules")
OUT_IPYNB = os.path.join(PROJECT_DIR, "SQUID_Manuscript_Figures.ipynb")


def read_module(name: str) -> str:
    with open(os.path.join(MODULES_DIR, name), "r") as f:
        src = f.read()
    # Strip the "if __name__ == '__main__':" standalone-test block, since
    # the notebook drives execution itself.
    marker = "\nif __name__ == \"__main__\":"
    idx = src.find(marker)
    if idx != -1:
        src = src[:idx].rstrip() + "\n"
    # __file__ is not defined inside a Jupyter cell; the notebook's setup
    # cell already puts modules/ on sys.path, so this per-module line is
    # both unnecessary and would raise NameError -- strip it.
    src = src.replace(
        "sys.path.insert(0, os.path.dirname(__file__))\n", ""
    )
    return src


nb = nbf.v4.new_notebook()
cells = []

# ==============================================================================
# TITLE
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell(r"""
### Contents
1. Environment setup & styling
2. Data loading & full numeric-value report
2b. Numerical-accuracy validation suite (independent recomputation checks)
3. Figure 1 -- SQUID noise-spectrum evolution across sensor lifetime
4. Figure 2 -- Quantum vs. classical anomaly-detection performance
5. Figure 3 -- Quantum Fisher information & entanglement-enhanced sensing
6. Figure 4 -- QPMS predictive-maintenance trajectory
7. Figure 5 -- VQC hyperparameter sweep (real parameter-shift training)
8. Figure 6 -- Graphical-abstract summary dashboard
9. Figure 7 -- Spectral feature-space analysis (correlation / PCA / t-SNE)
10. Figure 8 -- Monte Carlo uncertainty propagation & sensitivity analysis
11. Figure 9 -- Small-multiples gallery & rigorous statistical testing
12. Supplementary numeric-tables PDF & figure manifest
"""))

# ==============================================================================
# SECTION 1: SETUP
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("## 1. Environment setup & publication styling"))
cells.append(nbf.v4.new_code_cell(
    "import sys, os\n"
    "sys.path.insert(0, os.path.join(os.getcwd(), 'modules'))\n"
    "_repo_root = os.path.dirname(os.getcwd())  # notebooks/ -> repo root\n"
    "os.environ.setdefault('SQUID_DATA_DIR', os.path.join(_repo_root, 'data'))\n"
    "os.environ.setdefault('SQUID_FIGDIR', os.path.join(_repo_root, 'figures'))\n"
    "\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "from IPython.display import Image, display, Markdown\n"
))
cells.append(nbf.v4.new_code_cell(
    "import module_00_style_library as sl\n"
    "print('Style library loaded. Figures directory:', sl.FIGDIR)\n"
))

# ==============================================================================
# SECTION 2: DATA LOADING
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("## 2. Data loading, validation, and full numeric-value report"))
cells.append(nbf.v4.new_code_cell(read_module("module_01_data_loader.py")))
cells.append(nbf.v4.new_code_cell(
    "bundle = load_all_datasets()\n"
    "print_full_numeric_report(bundle)\n"
))

# ==============================================================================
# SECTION 2b: VALIDATION SUITE
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell(
    "## 2b. Numerical-accuracy validation suite\n\n"
    "Before any plotting, every key formula (GHZ QFI closed form, the "
    "visibility identity, ROC monotonicity, AUC bounds, the corner-frequency "
    "identity, and all probability/fraction ranges) is **independently "
    "re-derived from first principles** and checked against the saved CSV "
    "values -- this is the concrete mechanism behind this notebook's "
    "100%-accuracy requirement."
))
cells.append(nbf.v4.new_code_cell(read_module("module_02_validation_suite.py")))
cells.append(nbf.v4.new_code_cell("validation_result = run_validation_suite(bundle)\n"))


def add_figure_section(section_no: int, module_file: str, title: str, intro_md: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(f"## {section_no}. {title}\n\n{intro_md}"))
    cells.append(nbf.v4.new_code_cell(read_module(module_file)))
    var = f"fig{section_no}_path"
    cells.append(nbf.v4.new_code_cell(
        f"{var} = build_figure(bundle)\n"
        f"display(Image({var}.replace('.pdf', '.png'), width=920))\n"
    ))


add_figure_section(
    3, "module_fig01_noise_spectrum_evolution.py",
    "Figure 1 -- SQUID Noise-Spectrum Evolution Across Sensor Lifetime",
    r"""

"""
)

add_figure_section(
    4, "module_fig02_anomaly_detection_performance.py",
    "Figure 2 -- Quantum vs. Classical Anomaly-Detection Performance",
    r"""

"""
)

add_figure_section(
    5, "module_fig03_qfi_entanglement.py",
    "Figure 3 -- Quantum Fisher Information & Entanglement-Enhanced Sensing",
    r"""

"""
)

add_figure_section(
    6, "module_fig04_qpms_trajectory.py",
    "Figure 4 -- Quantum Predictive Maintenance Score (QPMS) Trajectory",
    r"""

"""
)

add_figure_section(
    7, "module_fig05_vqc_hyperparameter_sweep.py",
    "Figure 5 -- VQC Hyperparameter Sweep (Real Parameter-Shift-Rule Training)",
    r"""

"""
)

add_figure_section(
    8, "module_fig06_summary_dashboard.py",
    "Figure 6 -- Graphical-Abstract Summary Dashboard",
    r"""

"""
)

add_figure_section(
    9, "module_fig07_feature_correlation_analysis.py",
    "Figure 7 -- Spectral Feature-Space Analysis (Correlation / PCA / t-SNE)",
    r"""

"""
)

add_figure_section(
    10, "module_fig08_sensitivity_uncertainty.py",
    "Figure 8 -- Monte Carlo Uncertainty Propagation & Parameter Sensitivity",
    r"""

"""
)

add_figure_section(
    11, "module_fig09_small_multiples_gallery.py",
    "Figure 9 -- Small-Multiples Diagnostic Gallery & Rigorous Statistical Testing",
    r"""

"""
)

# ==============================================================================
# SECTION 10: MANIFEST + COMBINED PDF
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("## 12. Supplementary numeric-tables PDF & figure manifest"))
cells.append(nbf.v4.new_code_cell(read_module("module_10_numeric_tables_report.py")))
cells.append(nbf.v4.new_code_cell(
    "tables_pdf_path = build_report(bundle, validation_result)\n"
))
cells.append(nbf.v4.new_code_cell(
    "sl.print_figure_manifest()\n"
    "\n"
    "from pypdf import PdfWriter, PdfReader\n"
    "combined_path = os.path.join(sl.FIGDIR, 'ALL_FIGURES_COMBINED.pdf')\n"
    "writer = PdfWriter()\n"
    "for entry in sl.FIGURE_MANIFEST:\n"
    "    reader = PdfReader(entry.pdf_path)\n"
    "    for page in reader.pages:\n"
    "        writer.add_page(page)\n"
    "with open(combined_path, 'wb') as f:\n"
    "    writer.write(f)\n"
    "print(f'Combined {len(sl.FIGURE_MANIFEST)}-figure PDF booklet saved to: {combined_path}')\n"
))
cells.append(nbf.v4.new_markdown_cell(r"""

"""))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

with open(OUT_IPYNB, "w") as f:
    nbf.write(nb, f)

print(f"Notebook written to {OUT_IPYNB} with {len(cells)} cells")
