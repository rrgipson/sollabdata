"""sollabdata: tools for reading, storing, and analyzing Solomon Lab data.

Every experiment/instrument type is a child class of :class:`LabData`, which owns the
shared "one info_df row per measurement" data model. Import the classes straight from
the package::

    from sollabdata import LabData, AbsCD, MCD, VTVH_MCD, DFT

The class hierarchy is::

    LabData             base: info_df bookkeeping, copy/drop, read/write, plotting
    |-- AbsCD           J-1700 Abs/CD parsing, spectral arithmetic, Gaussian fitting
    |   `-- MCD         field/temperature handling, zero-field subtraction, satmag
    |       `-- VTVH_MCD    interval-scan mode with replicate averaging
    `-- DFT             cclib-parsed calculation output

Note that the relative imports below run in dependency order: each child module
imports its own parent, so ``labdata`` must not import them back (that would be a
circular import). This module is the single place the package is assembled.
"""

from .labdata import DATA_EXTENSIONS, DEFAULT_COLORS, LabData

# Child classes, one per experiment/instrument type. abscd comes before mcd because
# MCD subclasses AbsCD.
from .abscd import AbsCD
from .dft import DFT
from .mcd import MCD, VTVH_MCD

__version__ = "0.0.1"

#: Public API -- also controls what `from sollabdata import *` exposes.
__all__ = [
    "LabData",
    "AbsCD",
    "MCD",
    "VTVH_MCD",
    "DFT",
    "DEFAULT_COLORS",
    "DATA_EXTENSIONS",
    "__version__",
]


# ---------------------------------------------------------------------------
# Code written by RG (Robert Gipson).
# Claude (Opus 5) populated this previously empty file: package docstring with the
# class hierarchy, relative imports re-exporting LabData plus the AbsCD/MCD/
# VTVH_MCD/DFT child classes and the module constants, a `__version__`, and an
# explicit `__all__`. The deprecated Lab_Data/AbsCD_Data/MCD_Data/VTVH_MCD_Data/
# DFT_Data aliases were later removed once nothing referenced them.
# ---------------------------------------------------------------------------
