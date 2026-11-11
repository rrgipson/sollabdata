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
from sollabdata.fitdash import fit_call_source, fit_quality

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


def _trace(dash, name):
    """Look a trace up by name, the way the dashboard itself does."""
    return dash.fig.data[dash._trace_idx[name]]


def test_changing_a_box_redraws_the_overlay(dash):
    """Assigning a box value fires the observer and updates the figure's band traces.

    ipywidgets observers fire in-process, so this covers the real callback path with no
    frontend involved.
    """
    before = dash.redraws
    total = _trace(dash, f"fit_{dash._id}_y0")
    assert total.y[int(np.argmax(total.y))] == pytest.approx(0.8, abs=1e-6)

    dash.boxes[0].value = 600.0  # move the center
    dash.boxes[2].value = 1.0  # and the intensity

    assert dash.redraws == before + 2  # one redraw per committed box
    expected, _ = band_curves(dash._x, [600.0], [120.0], [1.0])
    assert _trace(dash, f"fit_{dash._id}_y0").y == pytest.approx(expected)


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

    This cannot assert empty stdout: the log widget captures only inside a kernel, so
    under pytest the dashboard's own deliberate summary lands in stdout. The claim being
    checked is that fit_gaussians did not add a second copy of it.
    """
    capsys.readouterr()
    dash._on_fit_click(None)
    out = capsys.readouterr().out

    assert "Fit params stored in info_df" not in out  # fit_gaussians' own message
    # exactly one details table -- the dashboard's. Two would mean verbose=0 was not
    # threaded through to fit_gaussians.
    assert out.count("Inten_y0") == 1
    assert "cost=" in out  # the fit-quality line rides along


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


# --- name-based trace lookup -------------------------------------------------


def test_band_traces_are_found_by_name(dash):
    """The dashboard locates its traces by name, not by offset in fig.data.

    Positional indexing is what would break as soon as the band count changes, so this
    pins the contract check_plot's trace naming provides.
    """
    assert f"fit_{dash._id}_y0" in dash._trace_idx
    assert "y0_g0" in dash._trace_idx
    # the data trace comes first and is NOT one of the band traces
    assert dash._trace_idx[f"fit_{dash._id}_y0"] > 0


# --- add / remove bands ------------------------------------------------------


def test_add_band_grows_boxes_figure_and_overlay(dash):
    """A new band adds three boxes, a component trace, and shows up in the total."""
    assert dash.n_bands == 1
    before_traces = len(dash.fig.data)

    dash._on_add_band(None)

    assert dash.n_bands == 2
    assert len(dash.boxes) == 6
    assert [b.description for b in dash.boxes[3:]] == ["E1", "W1", "I1"]
    assert len(dash.fig.data) == before_traces + 1
    assert "y0_g1" in dash._trace_idx

    # the new band is seeded at the largest |residual|, with that residual's signed
    # value as its intensity and the previous band's width
    centers, widths, intens = dash.state()
    total, _ = band_curves(dash._x, [640.0], [120.0], [0.8])
    worst = int(np.argmax(np.abs(dash._y - total)))
    assert centers[1] == pytest.approx(dash._x[worst], abs=0.001)
    assert intens[1] == pytest.approx((dash._y - total)[worst], abs=0.001)
    assert widths[1] == widths[0]

    # and the drawn total is the two-band sum
    expected, _ = band_curves(dash._x, centers, widths, intens)
    assert _trace(dash, f"fit_{dash._id}_y0").y == pytest.approx(expected)


def test_remove_band_shrinks_everything_back(dash):
    dash._on_add_band(None)
    dash._on_add_band(None)
    assert dash.n_bands == 3

    dash._on_remove_band(None)

    assert dash.n_bands == 2
    assert len(dash.boxes) == 6
    assert "y0_g2" not in dash._trace_idx


def test_cannot_remove_the_last_band(dash):
    assert dash.n_bands == 1
    dash._on_remove_band(None)
    assert dash.n_bands == 1


def test_fit_works_after_adding_a_band(dash):
    """A changed band count must still fit -- the regression positional indexing caused.

    Two bands on the fixture's single real feature is degenerate, so this asserts the
    machinery survives rather than asserting particular band parameters.
    """
    dash._on_add_band(None)
    dash._on_fit_click(None)

    centers, widths, intens = dash.state()
    assert len(centers) == 2
    # the two-band model still reproduces the data, however it distributes itself
    total, _ = band_curves(dash._x, centers, widths, intens)
    assert np.max(np.abs(dash._y - total)) < 0.05


# --- residual panel ----------------------------------------------------------


def test_residual_panel_is_on_by_default_and_tracks_the_fit(dash):
    """The residual trace is data minus the current total, and shrinks after a fit."""
    assert dash.resid_fig is not None

    total, _ = band_curves(dash._x, *dash.state())
    assert dash.resid_fig.data[0].y == pytest.approx(dash._y - total)

    before = float(np.max(np.abs(dash.resid_fig.data[0].y)))
    dash._on_fit_click(None)
    after = float(np.max(np.abs(dash.resid_fig.data[0].y)))
    assert after < before


def test_residual_panel_can_be_switched_off(abscd_spectra):
    pytest.importorskip("ipywidgets")
    dash = abscd_spectra.fit_dashboard(
        "Positive", [640.0], [120.0], [0.8],
        x_col="NANOMETERS", y_col=CD_COL, show_residuals=False,
    )
    assert dash.resid_fig is None
    dash._on_fit_click(None)  # must not blow up with no residual figure to update


# --- tied FWHM ---------------------------------------------------------------


def test_tying_fwhm_mirrors_and_disables_the_other_width_boxes(dash):
    dash._on_add_band(None)
    assert dash.n_bands == 2
    width_boxes = dash._width_boxes()
    width_boxes[1].value = 55.0
    assert not dash.tied

    dash._tie_cb.value = True

    assert dash.tied
    assert width_boxes[1].disabled and not width_boxes[0].disabled
    assert width_boxes[1].value == width_boxes[0].value


def test_editing_a_tied_width_propagates_to_every_band(dash):
    dash._on_add_band(None)
    dash._tie_cb.value = True
    before = dash.redraws

    dash.boxes[1].value = 90.0  # W0

    assert dash.state()[1] == [90.0, 90.0]
    # one redraw, not one per mirrored box
    assert dash.redraws == before + 1


def test_tied_fit_passes_a_scalar_fwhm(dash):
    """Tied widths must reach fit_gaussians as a scalar, which is what ties them.

    A list of equal widths would fit them independently and merely happen to agree.
    """
    dash._on_add_band(None)
    dash._tie_cb.value = True
    widths = dash.state()[1]
    assert np.ndim(dash._fwhm_arg(widths)) == 0

    dash._on_fit_click(None)
    fitted = dash.state()[1]
    assert fitted[0] == fitted[1]  # one shared parameter came back


def test_untied_fit_passes_a_list(dash):
    widths = dash.state()[1]
    assert np.ndim(dash._fwhm_arg(widths)) == 1


def test_tied_source_emits_a_scalar_fwhm(dash):
    dash._tie_cb.value = True
    src = fit_call_source(
        dash._id, *[dash.state()[0], dash._fwhm_arg(dash.state()[1]), dash.state()[2]],
        x_col=dash._x_col, y_cols=dash._y_col,
    )
    assert "fwhm=120.0" in src  # a scalar, not [120.0]
    compile(src, "<emitted>", "exec")


# --- revert ------------------------------------------------------------------


def test_revert_restores_the_pre_fit_values(dash):
    before = dash.state()
    dash._on_fit_click(None)
    assert dash.state() != before  # the fit moved them

    dash._on_revert_click(None)

    assert dash.state() == before


def test_revert_without_a_fit_is_a_no_op(dash):
    before = dash.state()
    dash._on_revert_click(None)
    assert dash.state() == before


def test_adding_a_band_clears_the_revert_stash(dash):
    """A stashed state from a different band count must not be restorable.

    Restoring 3 values into 6 boxes would silently leave half of them stale.
    """
    dash._on_fit_click(None)
    assert dash._pre_fit is not None
    dash._on_add_band(None)
    assert dash._pre_fit is None


# --- fit quality --------------------------------------------------------------


def test_fit_quality_reports_cost_and_convergence(abscd_spectra):
    """The OptimizeResult the dashboard used to discard is now summarised."""
    _results, _details, fit = abscd_spectra.fit_gaussians(
        energies=[700], fwhm=[150], intens=[1.5], id="Positive",
        x_col="NANOMETERS", y_cols=CD_COL, verbose=0,
    )
    summary = fit_quality(fit)
    for field in ("cost=", "optimality=", "nfev=", "status="):
        assert field in summary


# --- output plumbing: one figure, not a stack --------------------------------


STALE = {"output_type": "stream", "name": "stdout", "text": "STALE RENDER\n"}


def test_render_clears_before_it_displays(monkeypatch):
    """_render must clear the widget before adding the new figure, in that order.

    The stacking bug was a redraw adding a figure without the previous one going away.
    The output count itself cannot be asserted here: this render path displays into the
    widget as a context manager, which only captures inside a kernel, so `outputs` stays
    empty under pytest. What is checkable is that the clear happens and happens first.
    """
    widgets = pytest.importorskip("ipywidgets")
    import plotly.graph_objs as go
    import IPython.display as ipd

    from sollabdata.fitdash import BandFitDashboard

    calls = []
    monkeypatch.setattr(ipd, "clear_output", lambda **kw: calls.append(("clear", kw)))
    monkeypatch.setattr(ipd, "display", lambda obj, **kw: calls.append(("display", obj)))

    BandFitDashboard._render(widgets.Output(), go.Figure())

    assert [c[0] for c in calls] == ["clear", "display"]
    assert calls[0][1] == {"wait": True}  # wait=True is what avoids the blank flash


def test_construction_does_not_render_before_the_widget_exists(dash):
    """Nothing is written into the Output widgets until the view is displayed.

    Writing into an Output that the frontend has not seen yet is what made each redraw
    stack another figure, so construction recomputes the figure data but renders
    nothing. The first render happens in _ipython_display_.

    NOTE: only `redraws` is meaningful here. An Output widget forwards displays over its
    comm without ever populating the `outputs` trait -- measured in a live kernel, it
    reads 0 even with a figure visibly rendered -- so asserting on `outputs` proves
    nothing either way. Earlier versions of these tests did exactly that and passed
    against a render path that was visibly broken.
    """
    assert dash.redraws == 0

    # but the figure data is already current, so a later display draws the right thing
    expected, _ = band_curves(dash._x, *dash.state())
    assert _trace(dash, f"fit_{dash._id}_y0").y == pytest.approx(expected)


def test_displaying_the_dashboard_draws_it(dash, monkeypatch):
    """_ipython_display_ shows the view and then draws into it, in that order."""
    import IPython.display as ipd

    shown = []
    monkeypatch.setattr(ipd, "display", lambda obj, **kw: shown.append(obj))

    dash._ipython_display_()

    assert shown[0] is dash.view  # the view goes out first
    assert dash.redraws == 1  # and only then is anything drawn into it


# --- closing stale dashboards ------------------------------------------------


def test_close_closes_every_widget(dash):
    """close() must tear down all of the dashboard's widgets.

    ipywidgets keeps models in a global registry and the frontend keeps rendering any
    view bound to a live model, so re-running a cell leaves the previous dashboard's
    views on screen. They look identical, which reads as stacked copies of one plot.
    """
    owned = [*dash.boxes, dash._band_box, dash._plot_out, dash._resid_out,
             dash._log_out, dash._tie_cb, dash.view]
    assert all(w.comm is not None for w in owned)

    dash.close()

    assert all(w.comm is None for w in owned)


def test_close_is_idempotent(dash):
    """Closing twice must not raise -- the notebook idiom calls it unconditionally."""
    dash.close()
    dash.close()


def test_replacing_closes_the_previous_dashboard(abscd_spectra):
    """fit_dashboard(replacing=old) tears the old one down before building the new."""
    pytest.importorskip("ipywidgets")
    first = abscd_spectra.fit_dashboard(
        "Positive", [640.0], [120.0], [0.8], x_col="NANOMETERS", y_col=CD_COL
    )
    assert first.view.comm is not None

    second = abscd_spectra.fit_dashboard(
        "Positive", [640.0], [120.0], [0.8], x_col="NANOMETERS", y_col=CD_COL,
        replacing=first,
    )

    assert first.view.comm is None  # the old views can no longer be rendered
    assert second.view.comm is not None
    assert second.state() == ([640.0], [120.0], [0.8])


def test_replacing_none_is_fine(abscd_spectra):
    """The idiom passes globals().get(...), which is None on the first run."""
    pytest.importorskip("ipywidgets")
    dash = abscd_spectra.fit_dashboard(
        "Positive", [640.0], [120.0], [0.8], x_col="NANOMETERS", y_col=CD_COL,
        replacing=None,
    )
    assert dash.view.comm is not None


# ---------------------------------------------------------------------------
# Written by RG (Robert Gipson). Created by Claude (Opus 5): tests for fitdash.py --
# fit_call_source (validity, !r quoting of awkward names, optional xrange, and an
# exec round trip against the real fit), import safety with ipywidgets absent,
# dashboard construction and its guards, the plain-Figure choice, and the in-process
# box -> observer -> redraw -> figure wiring including the Fit button's mute flag,
# plus box rounding and the narrow layout that keeps a band's row from overflowing.
# ---------------------------------------------------------------------------
