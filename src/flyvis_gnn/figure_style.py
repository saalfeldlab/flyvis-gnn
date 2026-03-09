"""Centralized plot styling for the flyvis-gnn project.

Design principles:
  - flat design: no titles, no bounding box (all spines removed)
  - always lowercase text in labels and annotations
  - greek / LaTeX symbols via mathtext
  - consistent figure height (4 inches) across all plots
  - consistent font size (14pt base)
  - single DPI (200) for all saved figures
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Label constants — single source of truth for axis labels across the repo
# ---------------------------------------------------------------------------

LABEL_VOLTAGE = r"$v$"
LABEL_TAU = r"$\tau$"
LABEL_CALCIUM = r"$[\mathrm{Ca}^{2+}]$"
LABEL_TIME = "time (frames)"
LABEL_NEURONS = "neurons"
LABEL_FRAME = "frame"
LABEL_INPUT = "visual input"
LABEL_LOSS = "loss"
LABEL_EPOCH = "epoch"


@dataclass
class FigureStyle:
    """Centralized figure styling.

    Usage::

        from flyvis_gnn.figure_style import default_style as style

        style.apply_globally()          # once at program start
        fig, ax = style.figure()        # create pre-styled figure
        ax.plot(x, y)
        style.xlabel(ax, "time (frames)")
        style.savefig(fig, "out.png")
    """

    # --- typography --------------------------------------------------------
    font_family: str = "sans-serif"
    font_sans_serif: list[str] = field(
        default_factory=lambda: [
            "Nimbus Sans", "Arial", "Helvetica", "DejaVu Sans",
        ]
    )
    font_size: float = 14.0
    tick_font_size: float = 12.0
    label_font_size: float = 14.0
    annotation_font_size: float = 10.0
    use_latex: bool = False
    mathtext_fontset: str = "dejavusans"
    lowercase: bool = True

    # --- geometry ----------------------------------------------------------
    figure_height: float = 4.0          # inches — always the same
    default_aspect: float = 1.4         # width = height * aspect
    dpi: int = 200

    # --- colors ------------------------------------------------------------
    foreground: str = "black"
    background: str = "white"
    cmap: str = "viridis"
    cmap_calcium: str = "plasma"

    # --- decoration --------------------------------------------------------
    show_spines: bool = False           # flat design
    show_grid: bool = False
    grid_alpha: float = 0.3
    grid_color: str = "#e0e0e0"

    # --- line / marker defaults --------------------------------------------
    line_width: float = 1.2
    marker_size: float = 24.0

    # --- hex scatter defaults (spatial activity grids) ---------------------
    hex_marker_size: float = 72.0
    hex_stimulus_marker_size: float = 64.0
    hex_marker: str = "h"
    hex_voltage_range: tuple = (-2.0, 2.0)
    hex_stimulus_range: tuple = (0.0, 1.05)
    hex_calcium_range: tuple = (0.0, 2.0)

    # ---------------------------------------------------------------------- #
    #  Private helpers
    # ---------------------------------------------------------------------- #

    def _label(self, text: str) -> str:
        """Apply lowercase rule, preserving content inside $...$ math."""
        if not self.lowercase:
            return text
        import re
        parts = re.split(r'(\$[^$]*\$)', text)
        return ''.join(p if p.startswith('$') else p.lower() for p in parts)

    # ---------------------------------------------------------------------- #
    #  Public API
    # ---------------------------------------------------------------------- #

    def apply_globally(self) -> None:
        """Push style into matplotlib rcParams. Call once at program start."""
        plt.rcParams.update({
            "font.family": self.font_family,
            "font.sans-serif": self.font_sans_serif,
            "font.size": self.font_size,
            "axes.titlesize": self.font_size,
            "axes.labelsize": self.label_font_size,
            "xtick.labelsize": self.tick_font_size,
            "ytick.labelsize": self.tick_font_size,
            "legend.fontsize": self.tick_font_size,
            "text.usetex": self.use_latex,
            "mathtext.fontset": self.mathtext_fontset,
            "figure.facecolor": self.background,
            "axes.facecolor": self.background,
            "savefig.dpi": self.dpi,
            "savefig.pad_inches": 0.05,
            "figure.dpi": 100,
            "text.color": self.foreground,
            "axes.labelcolor": self.foreground,
            "xtick.color": self.foreground,
            "ytick.color": self.foreground,
        })

    def clean_ax(self, ax: Axes) -> Axes:
        """Apply flat design to a single axes: remove spines, tidy ticks."""
        for spine in ax.spines.values():
            spine.set_visible(self.show_spines)
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, color=self.grid_color)
        else:
            ax.grid(False)
        ax.tick_params(
            axis="both",
            which="both",
            labelsize=self.tick_font_size,
            colors=self.foreground,
        )
        return ax

    def figure(
        self,
        ncols: int = 1,
        nrows: int = 1,
        aspect: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        **subplot_kw,
    ) -> Tuple[Figure, Union[Axes, np.ndarray]]:
        """Create a pre-styled figure + axes.

        Height defaults to ``self.figure_height`` per row.
        Width is computed from aspect ratio unless overridden.

        Returns ``(fig, axes)`` where *axes* is a single ``Axes`` when
        ``nrows == ncols == 1``, otherwise a numpy array.
        """
        h = height or self.figure_height
        a = aspect or self.default_aspect
        w = width or (h * a * ncols / max(nrows, 1))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(w, h * nrows),
            facecolor=self.background,
            **subplot_kw,
        )

        if isinstance(axes, np.ndarray):
            for ax in axes.flat:
                self.clean_ax(ax)
        else:
            self.clean_ax(axes)

        return fig, axes

    def savefig(self, fig: Figure, path: str, close: bool = True, **kwargs) -> None:
        """Save with consistent DPI and tight bbox, then close."""
        defaults = dict(
            dpi=self.dpi,
            bbox_inches="tight",
            facecolor=self.background,
        )
        defaults.update(kwargs)
        fig.savefig(path, **defaults)
        if close:
            plt.close(fig)

    def xlabel(self, ax: Axes, text: str, **kwargs) -> None:
        """Set x-label with lowercase rule and consistent font."""
        ax.set_xlabel(
            self._label(text),
            fontsize=kwargs.pop("fontsize", self.label_font_size),
            color=kwargs.pop("color", self.foreground),
            **kwargs,
        )

    def ylabel(self, ax: Axes, text: str, **kwargs) -> None:
        """Set y-label with lowercase rule and consistent font."""
        ax.set_ylabel(
            self._label(text),
            fontsize=kwargs.pop("fontsize", self.label_font_size),
            color=kwargs.pop("color", self.foreground),
            **kwargs,
        )

    def annotate(self, ax: Axes, text: str, xy: tuple, **kwargs) -> None:
        """Add text annotation with consistent font."""
        ax.text(
            *xy,
            self._label(text),
            fontsize=kwargs.pop("fontsize", self.annotation_font_size),
            color=kwargs.pop("color", self.foreground),
            transform=kwargs.pop("transform", ax.transAxes),
            **kwargs,
        )


# --------------------------------------------------------------------------- #
#  Module-level singletons
# --------------------------------------------------------------------------- #

default_style = FigureStyle()
dark_style = FigureStyle(foreground="white", background="black")
