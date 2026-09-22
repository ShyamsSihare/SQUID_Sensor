from __future__ import annotations

import os
import string
import warnings
import datetime as _dt
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import gridspec
from matplotlib.colors import (
    LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm, to_rgba
)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    MaxNLocator, LogLocator, AutoMinorLocator, FuncFormatter, ScalarFormatter
)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ==============================================================================
# 1. GLOBAL CONSTANTS -- the manuscript's own visual identity
# ==============================================================================

MANUSCRIPT_TITLE = "Hybrid SQUID Sensor Arrays for Nuclear Plant Radiation Monitoring"
GENERATED_ON = _dt.date.today().isoformat()

#: Output directory for every PDF this suite produces.
try:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _THIS_DIR = os.getcwd()  # running inside a Jupyter cell -- no __file__
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
FIGDIR = os.environ.get("SQUID_FIGDIR", os.path.join(_REPO_ROOT, "figures"))
os.makedirs(FIGDIR, exist_ok=True)

#: Standard physics-journal column widths, in inches (APS / Nature / IEEE all
#: converge on these to within a few percent).
COLUMN_WIDTH_SINGLE_IN = 3.375   # single-column figure (PRX, PRL)
COLUMN_WIDTH_DOUBLE_IN = 7.00    # double-column / full-page-width figure
GOLDEN_RATIO = (1 + 5 ** 0.5) / 2

#: Standard DPI for rasterised elements embedded inside an otherwise-vector PDF
#: (e.g. large scatter clouds); journals typically require >= 300 for any
#: raster content, we use 600 for extra safety margin at print size.
RASTER_DPI = 600

# ------------------------------------------------------------------------------
# 1.1  Colour-blind-safe categorical palette (Okabe & Ito, 2008)
# ------------------------------------------------------------------------------
OKABE_ITO = {
    "black":        "#000000",
    "orange":       "#E69F00",
    "sky_blue":     "#56B4E9",
    "bluish_green": "#009E73",
    "yellow":       "#F0E442",
    "blue":         "#0072B2",
    "vermillion":   "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey":         "#999999",
}

#: Semantic role -> colour, used consistently across every figure so that
#: (for example) "quantum detector" is always the same blue everywhere in the
#: manuscript and "classical baseline" is always the same vermillion.
ROLE_COLOR = {
    "quantum":            OKABE_ITO["blue"],
    "quantum_alt":        OKABE_ITO["sky_blue"],
    "classical":          OKABE_ITO["vermillion"],
    "classical_alt":      OKABE_ITO["orange"],
    "healthy":            OKABE_ITO["bluish_green"],
    "anomaly":            OKABE_ITO["reddish_purple"],
    "threshold":          OKABE_ITO["black"],
    "neutral":            OKABE_ITO["grey"],
    "highlight":          OKABE_ITO["yellow"],
    "fresh":              OKABE_ITO["bluish_green"],
    "stage2":             OKABE_ITO["sky_blue"],
    "stage3":             OKABE_ITO["orange"],
    "stage4":             OKABE_ITO["vermillion"],
}

#: Failure-mode -> colour (five modes named in the manuscript Sec. 6.2)
FAILURE_MODE_COLORS = {
    "sensor_degradation": OKABE_ITO["vermillion"],
    "radiation_burst":    OKABE_ITO["reddish_purple"],
    "coolant_anomaly":    OKABE_ITO["blue"],
    "foreign_material":   OKABE_ITO["orange"],
    "calibration_drift":  OKABE_ITO["bluish_green"],
}
FAILURE_MODE_LABELS = {
    "sensor_degradation": "Sensor degradation",
    "radiation_burst":    "Radiation burst",
    "coolant_anomaly":    "Coolant flow anomaly",
    "foreign_material":   "Foreign material",
    "calibration_drift":  "Calibration drift",
}

# ------------------------------------------------------------------------------
# 1.2  Perceptually-uniform, colour-blind-safe continuous colormaps
# ------------------------------------------------------------------------------
CMAP_SEQUENTIAL = "viridis"       # general-purpose sequential (perceptually uniform)
CMAP_SEQUENTIAL_ALT = "cividis"   # colour-blind-optimised alternative
CMAP_DIVERGING = "RdBu_r"         # diverging, colour-blind safe
CMAP_DAMAGE = LinearSegmentedColormap.from_list(
    "damage_progression",
    ["#009E73", "#F0E442", "#E69F00", "#D55E00"],  # green -> yellow -> orange -> red
    N=256,
)
CMAP_QUANTUM_CLASSICAL = LinearSegmentedColormap.from_list(
    "quantum_classical_diverging",
    ["#D55E00", "#F7F7F7", "#0072B2"], N=256,
)


# ==============================================================================
# 2. GLOBAL RCPARAMS -- journal-grade typography & line-work
# ==============================================================================

def _find_best_serif_font() -> str:
    preferred = [
        "Latin Modern Roman", "CMU Serif", "Times New Roman",
        "Nimbus Roman", "Liberation Serif", "STIX Two Text", "DejaVu Serif",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return "DejaVu Serif"


SERIF_FONT = _find_best_serif_font()


def apply_journal_style(base_fontsize: int = 8.5) -> None:
    mpl.rcdefaults()
    plt.rcParams.update({
        # ---- Fonts -------------------------------------------------------
        "font.family": "serif",
        "font.serif": [SERIF_FONT],
        "mathtext.fontset": "cm",
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 1,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1.5,
        "figure.titlesize": base_fontsize + 3,
        # ---- Vector / embedding safety ------------------------------------
        "pdf.fonttype": 42,     # embed as TrueType, not bitmap Type-3
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 120,       # on-screen preview only; save-time DPI is separate
        "savefig.dpi": RASTER_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
        # ---- Axes / spines -------------------------------------------------
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#111111",
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.unicode_minus": True,
        # ---- Ticks -----------------------------------------------------
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # ---- Lines / markers ---------------------------------------------
        "lines.linewidth": 1.3,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 0.6,
        "errorbar.capsize": 2.2,
        "patch.linewidth": 0.7,
        # ---- Legend ------------------------------------------------------
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#888888",
        "legend.fancybox": False,
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.6,
        "legend.labelspacing": 0.35,
        # ---- Figure ------------------------------------------------------
        "figure.constrained_layout.use": False,  # we hand-tune GridSpec margins
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


# ==============================================================================
# 3. PANEL-LABELLING & ANNOTATION HELPERS
# ==============================================================================

def panel_letters(n: int) -> list[str]:
    letters = list(string.ascii_lowercase)
    if n <= 26:
        return letters[:n]
    out = []
    i = 0
    while len(out) < n:
        out.append(letters[i % 26] + (str(i // 26 + 1) if i >= 26 else ""))
        i += 1
    return out


def label_panel(ax: plt.Axes, letter: str, x: float = -0.16, y: float = 1.06,
                 fontsize: Optional[float] = None, weight: str = "bold") -> None:
    fontsize = fontsize or (plt.rcParams["font.size"] + 2)
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fontsize,
             fontweight=weight, va="bottom", ha="right", family=SERIF_FONT)


def add_significance_bracket(ax: plt.Axes, x1: float, x2: float, y: float,
                              text: str, height: float = 0.02,
                              linewidth: float = 0.8, fontsize: float = 7.0) -> None:
    y2 = y + height
    ax.plot([x1, x1, x2, x2], [y, y2, y2, y], lw=linewidth, c="black",
             transform=ax.get_xaxis_transform() if False else ax.transData, clip_on=False)
    ax.text((x1 + x2) / 2, y2, text, ha="center", va="bottom", fontsize=fontsize)


def inset_zoom(ax: plt.Axes, bounds: Sequence[float],
                xlim: Sequence[float], ylim: Sequence[float],
                edgecolor: str = "#444444") -> plt.Axes:
    axins = ax.inset_axes(bounds)
    axins.set_xlim(*xlim)
    axins.set_ylim(*ylim)
    axins.tick_params(labelsize=6, direction="in")
    for spine in axins.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor(edgecolor)
    ax.indicate_inset_zoom(axins, edgecolor=edgecolor, linewidth=0.7, alpha=0.9)
    return axins


def styled_colorbar(mappable, ax_or_cax, label: str, fig: plt.Figure,
                     orientation: str = "vertical", shrink: float = 0.9,
                     pad: float = 0.02, ticks: Optional[Sequence[float]] = None):
    cb = fig.colorbar(mappable, ax=ax_or_cax, orientation=orientation,
                       shrink=shrink, pad=pad, ticks=ticks)
    cb.set_label(label, fontsize=plt.rcParams["font.size"] - 0.5)
    cb.ax.tick_params(labelsize=plt.rcParams["font.size"] - 2, direction="in")
    cb.outline.set_linewidth(0.6)
    return cb


def watermark_provenance(fig: plt.Figure, note: str) -> None:
    fig.text(0.005, 0.002, note, fontsize=4.6, color="#999999",
              family=SERIF_FONT, ha="left", va="bottom", style="italic")


# ==============================================================================
# 4. FIGURE-SAVING PIPELINE
# ==============================================================================

@dataclass
class FigureManifestEntry:
    fig_id: str
    title: str
    n_panels: int
    pdf_path: str
    n_data_points: int = 0
    notes: str = ""


FIGURE_MANIFEST: list[FigureManifestEntry] = []


def save_publication_figure(fig: plt.Figure, fig_id: str, title: str,
                             n_panels: int, n_data_points: int = 0,
                             notes: str = "", also_png: bool = True) -> str:
    watermark_provenance(
        fig,
        f"Simulated data (not hardware/field measurements) - {MANUSCRIPT_TITLE} - "
        f"generated {GENERATED_ON} - {fig_id}"
    )
    pdf_path = os.path.join(FIGDIR, f"{fig_id}.pdf")
    metadata = {
        "Title": title,
        "Author": "Numerical simulation pipeline (see README.md)",
        "Subject": MANUSCRIPT_TITLE,
        "Keywords": "SQUID, quantum sensing, radiation monitoring, VQC, anomaly detection",
        "CreationDate": _dt.datetime.now(),
    }
    fig.savefig(pdf_path, format="pdf", metadata=metadata)
    if also_png:
        fig.savefig(os.path.join(FIGDIR, f"{fig_id}.png"), format="png", dpi=RASTER_DPI)
    FIGURE_MANIFEST.append(FigureManifestEntry(
        fig_id=fig_id, title=title, n_panels=n_panels, pdf_path=pdf_path,
        n_data_points=n_data_points, notes=notes,
    ))
    print(f"  [saved] {fig_id}.pdf  ({n_panels} panels, {n_data_points:,} data points)  -> {pdf_path}")
    return pdf_path


def print_figure_manifest() -> None:
    print("\n" + "=" * 88)
    print("FIGURE MANIFEST")
    print("=" * 88)
    header = f"{'ID':<10}{'Panels':>7}{'DataPts':>12}   Title"
    print(header)
    print("-" * 88)
    for e in FIGURE_MANIFEST:
        print(f"{e.fig_id:<10}{e.n_panels:>7}{e.n_data_points:>12,}   {e.title}")
    print("=" * 88 + f"\nTotal figures: {len(FIGURE_MANIFEST)}\n")


# ==============================================================================
# 5. NUMERIC-VALUE REPORTING HELPERS ("print the numerical values used")
# ==============================================================================

def print_section_header(title: str) -> None:
    bar = "=" * 88
    print(f"\n{bar}\n{title}\n{bar}")


def print_numeric_summary(df, columns: Optional[Sequence[str]] = None,
                           label: str = "") -> None:
    import pandas as pd  # local import keeps this module import-light
    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if label:
        print(f"  -- {label} --")
    stats = df[cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    stats = stats.rename(columns={"50%": "median"})
    with mpl.rc_context({"font.size": 8}):
        pass
    print(stats.to_string(float_format=lambda v: f"{v:.5g}"))


def fmt_sci(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}g}"


# ==============================================================================
# 6. FIGURE-SIZE PRESETS
# ==============================================================================

def figsize_grid(ncols: int, nrows: int, panel_w: float = 2.15,
                  panel_h: float = 1.85) -> tuple[float, float]:
    return (panel_w * ncols + 0.6, panel_h * nrows + 0.55)


# Apply the style immediately on import so every downstream module inherits it.
apply_journal_style()

print(f"[module_00_style_library] journal style applied "
      f"(serif={SERIF_FONT!r}, figdir={FIGDIR!r})")
