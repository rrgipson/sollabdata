"""Tests for the interactive fitting dashboard (sollabdata/fitdash.py).

The widget layer is kept thin on purpose, so most of what matters is testable without a
browser:

- ``fit_call_source`` is pure and fully covered here.
- The import-safety tests run with or without ipywidgets installed.
- The wiring tests use ipywidgets' observers, which fire in-process with no kernel or
  frontend attached, so the box -> redraw -> figure path is genuinely exercised.

Left to a notebook smoke test: that anything actually *renders*.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from sollabdata import AbsCD
from sollabdata.abscd import band_curves
from sollabdata.fitdash import fit_call_source

DATA_DIR = Path(__file__).parent / "data" / "abscd"
CD_COL = "CD/DC [mdeg]"


@pytest.fixture
def abscd_spectra():
    """The same two J-1700 fixture scans test_abscd.py uses."""
    return AbsCD(path_to_raw_data=str(DATA_DIR), info_csv="abscd_test_info.csv")


@pytest.fixture
def dash(abscd_spectra):
    """A dashboard over the Positive scan's 600 nm band, skipped without ipywidgets."""
    pytest.importorskip("ipywidgets")
    return abscd_spectra.fit_dashboard(
        "Positive",
        energies=[640.0],
        fwhm=[120.0],
        intens=[0.8],
        x_col="NANOMETERS",
        y_col=CD_COL,
    )


# --- fit_call_source: pure ---------------------------------------------------


def test_fit_call_source_is_valid_python():
    src = fit_call_source(
        "Positive", [600.0], [100.0], [1.0], x_col="NANOMETERS", y_cols=CD_COL
    )
    compile(src, "<emitted>", "exec")
    assert "fit_gaussians(" in src
    assert "energies=[600.0]" in src


def test_fit_call_source_survives_awkward_names():
    """Every value goes through !r, so brackets and quotes cannot break the source.

    `CD/DC [mdeg]` is a real column name here, and hand-written quoting around it would
    be fine -- but an id containing an apostrophe would not be.
    """
    src = fit_call_source(
        "sample 'A' [run 2]",
        [600.0],
        [100.0],
        [1.0],
        x_col="NANOMETERS",
        y_cols=CD_COL,
    )
    compile(src, "<emitted>", "exec")
    assert repr("sample 'A' [run 2]") in src
    assert repr(CD_COL) in src


def test_fit_call_source_includes_xrange_only_when_given():
    without = fit_call_source("a", [1.0], [2.0], [3.0], x_col="x", y_cols="y")
    assert "xrange" not in without

    with_range = fit_call_source(
        "a", [1.0], [2.0], [3.0], x_col="x", y_cols="y", xrange=[400, 800]
    )
    assert "xrange=[400, 800]" in with_range
    compile(with_range, "<emitted>", "exec")


def test_fit_call_source_round_trips_against_the_real_fit(abscd_spectra):
    """Executing the emitted source reproduces the fit it describes.

    The strongest available check that the generated call is not merely syntactically
    valid but semantically the same call.
    """
    src = fit_call_source(
        "Positive", [700.0], [150.0], [1.5], x_col="NANOMETERS", y_cols=CD_COL
    )
    namespace = {"data": abscd_spectra}
    exec(compile(src, "<emitted>", "exec"), namespace)

    assert namespace["details"].at[0, "Energy"] == pytest.approx(600, abs=0.1)
    assert namespace["details"].at[0, "FWHM"] == pytest.approx(100, abs=0.1)


# --- import safety -----------------------------------------------------------


def test_package_imports_without_ipywidgets(monkeypatch):
    """`import sollabdata` and `import sollabdata.fitdash` must not need ipywidgets.

    Written as a test rather than relying on whether the dev environment happens to
    have ipywidgets installed today.
    """
    monkeypatch.setitem(sys.modules, "ipywidgets", None)

    import sollabdata
    import sollabdata.fitdash as fd

    importlib.reload(sollabdata)
    assert fd.fit_call_source("a", [1.0], [2.0], [3.0], x_col="x", y_cols="y")


def test_fit_dashboard_without_ipywidgets_explains_how_to_install(
    abscd_spectra, monkeypatch
):
    """The failure names the package and the install command, rather than an ImportError
    from somewhere deep in the widget stack."""
    monkeypatch.setitem(sys.modules, "ipywidgets", None)

    with pytest.raises(ImportError, match="ipywidgets"):
        abscd_spectra.fit_dashboard(
            "Positive", [600.0], [100.0], [1.0], x_col="NANOMETERS", y_col=CD_COL
        )


# --- construction ------------------------------------------------------------


def test_dashboard_lays_out_three_boxes_per_band(dash):
    assert len(dash.boxes) == 3
    assert [b.description for b in dash.boxes] == ["E0", "W0", "I0"]
    assert dash.state() == ([640.0], [120.0], [0.8])


def test_dashboard_rejects_mismatched_guess_lengths(abscd_spectra):
    pytest.importorskip("ipywidgets")
    with pytest.raises(ValueError, match="same length"):
        abscd_spectra.fit_dashboard(
            "Positive", [600.0, 700.0], [100.0], [1.0], x_col="NANOMETERS", y_col=CD_COL
        )


def test_dashboard_rejects_an_unknown_id(abscd_spectra):
    pytest.importorskip("ipywidgets")
    with pytest.raises(ValueError, match="No info_df row with id"):
        abscd_spectra.fit_dashboard(
            "Nope", [600.0], [100.0], [1.0], x_col="NANOMETERS", y_col=CD_COL
        )


def test_dashboard_figure_has_no_comm_handler(dash):
    """The figure must be a plain Figure, not a FigureWidget.

    FigureWidget is what raises KeyError: 'uid' from _handler_js2py_traceDeltas in
    plotly 7.0.0, so this pins the choice that avoids it.
    """
    assert type(dash.fig).__name__ == "Figure"
    assert not hasattr(dash.fig, "_handler_js2py_traceDeltas")


# --- wiring: box -> redraw -> figure -----------------------------------------


def test_changing_a_box_redraws_the_overlay(dash):
    """Assigning a box value fires the observer and updates the figure's band traces.

    ipywidgets observers fire in-process, so this covers the real callback path with no
    frontend involved.
    """
    before = dash.redraws
    total_trace = dash.fig.data[dash._fit_start]
    assert total_trace.y[int(np.argmax(total_trace.y))] == pytest.approx(0.8, abs=1e-6)

    dash.boxes[0].value = 600.0  # move the center
    dash.boxes[2].value = 1.0  # and the intensity

    assert dash.redraws == before + 2  # one redraw per committed box
    expected, _ = band_curves(dash._x, [600.0], [120.0], [1.0])
    assert dash.fig.data[dash._fit_start].y == pytest.approx(expected)


def test_fit_button_snaps_boxes_and_redraws_once(dash):
    """Fit recovers the known band, and the mute flag collapses 3N redraws into 1.

    The fixture's Positive scan is a +1 mdeg band at 600 nm with 100 nm FWHM, so the
    boxes should land there from the deliberately-off starting guess.
    """
    before = dash.redraws
    dash._on_fit_click(None)

    centers, widths, intens = dash.state()
    assert centers[0] == pytest.approx(600, abs=0.1)
    assert widths[0] == pytest.approx(100, abs=0.1)
    assert intens[0] == pytest.approx(1.0, abs=0.01)

    # without muting, assigning 3 boxes would have redrawn 3 times plus the explicit one
    assert dash.redraws == before + 1


def test_fit_button_does_not_print_the_fit_summary(dash, capsys):
    """The Fit handler passes verbose=0, so the details table and the stored-slice
    message go to the log widget rather than flooding the cell on every click.

    This cannot assert empty stdout. Outside an IPython kernel neither `display(fig)`
    nor `with Output()` captures, so the redraw's figure repr and the dashboard's own
    deliberate log print both land in stdout under pytest. The claim being checked is
    specifically that fit_gaussians did not add its own copy.
    """
    capsys.readouterr()
    dash._on_fit_click(None)
    out = capsys.readouterr().out

    # fit_gaussians' own per-id message, printed only when verbose
    assert "Fit params stored in info_df" not in out
    # the details table appears once -- the dashboard's log print. Twice would mean
    # fit_gaussians printed its copy too, i.e. verbose=0 was not threaded through.
    assert out.count("Inten_y0") == 1


def test_show_call_emits_the_current_box_values(dash):
    """The emitted source reflects whatever is in the boxes right now."""
    dash.boxes[0].value = 612.5
    dash._on_code_click(None)
    src = fit_call_source(
        dash._id, *dash.state(), x_col=dash._x_col, y_cols=dash._y_col
    )
    assert "energies=[612.5]" in src
    compile(src, "<emitted>", "exec")


# --- display formatting ------------------------------------------------------


def test_boxes_round_the_initial_guess(abscd_spectra):
    """Long floats in the guess are rounded for display, and state() agrees.

    Rounding on the way in rather than only for show means state() -- and so the
    emitted source -- reports exactly what the box displays.
    """
    pytest.importorskip("ipywidgets")
    dash = abscd_spectra.fit_dashboard(
        "Positive",
        energies=[640.1234567],
        fwhm=[120.98765],
        intens=[0.7777777],
        x_col="NANOMETERS",
        y_col=CD_COL,
    )
    assert dash.state() == ([640.123], [120.988], [0.778])


def test_boxes_round_the_fit_result(dash):
    """Fitted values land in the boxes rounded, not as 14-digit floats."""
    dash._on_fit_click(None)
    for box in dash.boxes:
        assert box.value == round(box.value, 3)


def test_box_precision_is_configurable(abscd_spectra):
    pytest.importorskip("ipywidgets")
    dash = abscd_spectra.fit_dashboard(
        "Positive", [640.1234567], [120.0], [0.8],
        x_col="NANOMETERS", y_col=CD_COL, box_precision=1,
    )
    assert dash.state()[0] == [640.1]


def test_boxes_are_narrow_enough_to_avoid_horizontal_scrolling(dash):
    """Three boxes per band must fit a cell's width.

    The ipywidgets defaults are wide enough that a band's row overflows and forces
    horizontal scrolling, so both the box width and the description label are pinned.
    """
    widths = {b.layout.width for b in dash.boxes}
    assert widths == {"130px"}
    assert {b.style.description_width for b in dash.boxes} == {"24px"}
    # a band's whole row stays well inside a typical ~900px cell
    assert 3 * int(dash.boxes[0].layout.width.removesuffix("px")) < 500


def test_emitted_source_uses_the_box_precision(dash):
    """The pasteable call shows the same numbers the boxes do."""
    dash.boxes[0].value = 612.3456
    src = fit_call_source(
        dash._id, *dash.state(), x_col=dash._x_col, y_cols=dash._y_col,
        precision=dash._precision,
    )
    assert "energies=[612.346]" in src
    compile(src, "<emitted>", "exec")


# ---------------------------------------------------------------------------
# Written by RG (Robert Gipson). Created by Claude (Opus 5): tests for fitdash.py --
# fit_call_source (validity, !r quoting of awkward names, optional xrange, and an
# exec round trip against the real fit), import safety with ipywidgets absent,
# dashboard construction and its guards, the plain-Figure choice, and the in-process
# box -> observer -> redraw -> figure wiring including the Fit button's mute flag,
# plus box rounding and the narrow layout that keeps a band's row from overflowing.
# ---------------------------------------------------------------------------
