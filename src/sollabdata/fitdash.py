"""Interactive Gaussian band fitting for notebooks.

A small dashboard for the step that actually costs time in band fitting: landing on a
good initial guess. Numeric boxes hold each band's center, width and intensity; the
model curve redraws over the data whenever a value is committed; bands can be added and
removed; widths can be tied to one shared parameter; a Fit button runs the real
:meth:`~sollabdata.abscd.AbsCD.fit_gaussians` from the current values and snaps the boxes
to the answer, with Revert to undo it; a residual panel shows whether what is left over
is structureless; and a Show-call button emits the equivalent ``fit_gaussians(...)``
source to paste into a cell, so the notebook stays the record of what was done.

Rendering uses an ``ipywidgets.Output`` holding a plain plotly ``Figure`` rather than a
``FigureWidget``. That is deliberate: plotly 7.0.0's ``FigureWidget`` raises
``KeyError: 'uid'`` in ``BaseFigureWidget._handler_js2py_traceDeltas`` when the frontend
echoes back a trace delta without a uid, which poisons the kernel's message handler and
survives a kernel restart. A plain ``Figure`` has no comm channel, so that code path does
not exist. The cost is that each update re-renders rather than mutating in place, which
is why the boxes commit on Enter instead of streaming like a slider.

ipywidgets is imported lazily, inside the entry point, so ``import sollabdata`` works
without it. Install with ``pip install -e ".[widgets]"``, or in a conda environment
``conda install -c conda-forge ipywidgets``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import plotly.graph_objs as go

from .abscd import WAVE, band_curves

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .abscd import AbsCD


def _require_ipywidgets():
    """Import ipywidgets, or raise with the install command.

    Deferred so the package imports fine without a notebook stack installed.
    """
    try:
        import ipywidgets
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(
            "fit_dashboard needs ipywidgets. Install it with "
            'pip install -e ".[widgets]", or in a conda environment with '
            "conda install -c conda-forge ipywidgets"
        ) from exc
    return ipywidgets


def fit_call_source(
    id,
    energies: Sequence[float],
    fwhm,
    intens: Sequence[float],
    x_col: str,
    y_cols,
    xrange=None,
    var_name: str = "data",
    precision: int = 4,
) -> str:
    """Render the equivalent ``fit_gaussians(...)`` call as pasteable source.

    Follows the ``LabData.prep_plt`` idiom of emitting code for the user to paste and
    edit, but returns the string rather than printing it, so it can be unit-tested and
    so the caller decides where it goes.

    A scalar ``fwhm`` is emitted as a scalar, matching the convention fit_gaussians uses
    to tie every band to one shared width.

    Every value is interpolated with ``!r``, which is what keeps a column name like
    ``"CD/DC [mdeg]"`` -- or an id containing a quote or bracket -- from producing
    broken source.
    """

    def nums(values):
        return [round(float(v), precision) for v in values]

    fwhm_arg = (
        round(float(fwhm), precision) if np.ndim(fwhm) == 0 else nums(fwhm)
    )
    lines = [
        f"# {var_name} = YOUR_OBJECT",
        f"results, details, fit = {var_name}.fit_gaussians(",
        f"    energies={nums(energies)!r},",
        f"    fwhm={fwhm_arg!r},",
        f"    intens={nums(intens)!r},",
        f"    id={id!r},",
        f"    x_col={x_col!r},",
        f"    y_cols={y_cols!r},",
    ]
    if xrange is not None:
        lines.append(f"    xrange={list(xrange)!r},")
    lines.append(")")
    return "\n".join(lines)


def fit_quality(fit) -> str:
    """One-line summary of a scipy OptimizeResult from fit_gaussians.

    ``cost`` is 0.5 * sum of squared residuals and ``optimality`` the first-order
    optimality measure, so together they say whether the fit actually converged or
    merely stopped. The dashboard previously discarded this entirely.
    """
    return (
        f"cost={fit.cost:.4g}  optimality={fit.optimality:.3g}  "
        f"nfev={fit.nfev}  status={fit.status} ({fit.message})"
    )


class BandFitDashboard:
    """Boxes + live overlay + Fit button for one spectrum's Gaussian bands.

    Prefer :meth:`sollabdata.abscd.AbsCD.fit_dashboard`, which constructs this.

    Scope is deliberately one id and one y column. Multiple ids or y columns are
    supported by ``fit_gaussians`` itself and are clearer in pasted source.

    Band traces are located by *name* rather than by position in ``fig.data``.
    ``check_plot`` names them ``fit_<id>_y0`` and ``y0_g<k>``, which is what lets the
    band count change at runtime without the indexing going wrong.

    Attributes:
        fig: the plotly Figure being redrawn.
        resid_fig: the residual Figure, or None when show_residuals is False.
        boxes: the FloatText widgets, laid out ``[E0, W0, I0, E1, W1, I1, ...]``.
        redraws: how many redraws have happened, which makes the mute flag's effect
            observable from a test.

    Args to __init__ beyond the obvious:
        show_residuals: draw a residual panel (data minus total fit) under the main
            figure. Structureless residuals are how a band fit is judged, so this
            defaults on; pass False to save the vertical space.
        tie_fwhm: start with every band's width tied to one shared parameter, as a
            scalar ``fwhm`` does in fit_gaussians. Toggleable from the checkbox.
        box_precision: decimal places shown in the boxes, and used for the emitted
            source. Values are rounded on the way in, so ``state()`` returns what is
            displayed rather than a longer float the box is truncating for show.
        box_width: CSS width of each box. The default fits ~10 digits while keeping
            three boxes on one line without horizontal scrolling.
    """

    def __init__(
        self,
        data: "AbsCD",
        id,
        energies: Sequence[float],
        fwhm: Sequence[float],
        intens: Sequence[float],
        x_col: str = WAVE,
        y_col: str = "deps",
        xrange=None,
        var_name: str = "data",
        show_residuals: bool = True,
        tie_fwhm: bool = False,
        box_precision: int = 3,
        box_width: str = "130px",
        **plot_kwargs,
    ) -> None:
        widgets = _require_ipywidgets()
        self._w = widgets

        energies = [float(v) for v in energies]
        fwhm = [float(v) for v in fwhm]
        intens = [float(v) for v in intens]
        if not (len(fwhm) == len(intens) == len(energies)):
            raise ValueError(
                f"energies, fwhm and intens must be the same length, got "
                f"{len(energies)}, {len(fwhm)} and {len(intens)}."
            )
        if not energies:
            raise ValueError("Pass at least one band.")

        self._data = data
        self._id = id
        self._x_col = x_col
        self._y_col = y_col
        self._xrange = xrange
        self._var_name = var_name
        self._precision = box_precision
        self._box_width = box_width
        self._show_residuals = show_residuals
        self._muted = False
        self._pre_fit = None
        self.redraws = 0

        # The x/y data the overlay and residuals are computed against, windowed the
        # same way check_plot windows its own.
        row = data.info_df.loc[data.info_df["id"] == id]
        if not len(row):
            raise ValueError(f"No info_df row with id {id!r}.")
        frame = row["data"].iloc[0]
        if xrange is not None:
            frame = frame.loc[(frame[x_col] > xrange[0]) & (frame[x_col] < xrange[1])]
        self._x = frame[x_col].to_numpy()
        self._y = frame[y_col].to_numpy()
        self._plot_kwargs = plot_kwargs

        # Containers whose children are reassigned when the band count changes, so the
        # view object the notebook is holding stays the same one.
        self._band_box = widgets.VBox([])
        self._plot_out = widgets.Output()
        self._resid_out = widgets.Output()
        self._log_out = widgets.Output()
        self._tie_cb = widgets.Checkbox(
            value=bool(tie_fwhm),
            description="Tie FWHM",
            indent=False,
            layout=widgets.Layout(width="110px"),
        )
        self._tie_cb.observe(self._on_tie_change, names="value")

        self.resid_fig = self._make_resid_fig() if show_residuals else None

        panels = [self._band_box, self._controls(), self._plot_out]
        if show_residuals:
            panels.append(self._resid_out)
        panels.append(self._log_out)
        self.view = widgets.VBox(panels)

        self._rebuild(energies, fwhm, intens)

    # -- construction helpers ----------------------------------------------

    def _controls(self):
        widgets = self._w
        buttons = [
            ("Fit", "primary", self._on_fit_click),
            ("Revert", "", self._on_revert_click),
            ("+ band", "", self._on_add_band),
            ("- band", "", self._on_remove_band),
            ("Show call", "", self._on_code_click),
        ]
        row = []
        for label, style, handler in buttons:
            btn = widgets.Button(
                description=label,
                button_style=style,
                layout=widgets.Layout(width="90px"),
            )
            btn.on_click(handler)
            row.append(btn)
        row.append(self._tie_cb)
        return widgets.HBox(row)

    def _make_resid_fig(self):
        """A separate small figure for residuals.

        Separate rather than another row in check_plot's subplot grid: an extra row
        would shift every band trace, and keeping it out means check_plot needs no
        knowledge of the dashboard at all.
        """
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self._x, y=np.zeros_like(self._x), mode="lines", name="residual"
            )
        )
        fig.add_hline(y=0, line=dict(width=1, dash="dot"))
        fig.update_layout(
            height=180,
            width=self._plot_kwargs.get("width", 1000),
            margin=dict(t=30, b=40, l=20),
            title="residual (data - fit)",
            xaxis_title=self._x_col,
        )
        if self._xrange is not None:
            fig.update_xaxes(range=self._xrange)
        return fig

    def _make_boxes(self, energies, fwhm, intens):
        """One row of three unbounded FloatTexts per band.

        Unbounded on purpose: BoundedFloatText silently clamps an out-of-range
        assignment, so a fit landing outside a box's range would display -- and emit as
        pasteable source -- a number that is not the fit result.
        """
        widgets = self._w
        # A narrow box plus a narrow description label: the ipywidgets defaults are wide
        # enough that three boxes overflow the cell and force horizontal scrolling.
        layout = widgets.Layout(width=self._box_width)
        style = {"description_width": "24px"}
        rows, boxes = [], []
        for i, (c, w, v) in enumerate(zip(energies, fwhm, intens)):
            trio = [
                widgets.FloatText(
                    value=self._round(c),
                    description=f"E{i}",
                    step=10.0,
                    layout=layout,
                    style=style,
                ),
                widgets.FloatText(
                    value=self._round(w),
                    description=f"W{i}",
                    step=10.0,
                    layout=layout,
                    style=style,
                ),
                widgets.FloatText(
                    value=self._round(v),
                    description=f"I{i}",
                    step=0.1,
                    layout=layout,
                    style=style,
                ),
            ]
            boxes += trio
            rows.append(widgets.HBox(trio))
        return boxes, rows

    def _round(self, value) -> float:
        """Round to the configured precision, so the boxes do not show noise digits."""
        return round(float(value), self._precision)

    def _rebuild(self, energies, fwhm, intens):
        """Rebuild boxes and figure for a (possibly new) band count, then redraw.

        Called at construction and whenever a band is added or removed. The figure has
        to be rebuilt because its trace count changes with the band count.
        """
        self.boxes, rows = self._make_boxes(energies, fwhm, intens)
        for box in self.boxes:
            box.observe(self._on_param_change, names="value")
        self._band_box.children = tuple(rows)

        self.fig = self._data.check_plot(
            self._id,
            result=self._data._pack_fitvars(energies, fwhm, [intens]),
            x_col=self._x_col,
            y_cols=self._y_col,
            xrange=self._xrange,
            **self._plot_kwargs,
        )
        # Name -> position, rebuilt with the figure. check_plot labels the total fit
        # `fit_<id>_y0` and each component `y0_g<k>`. Looking traces up by name rather
        # than by offset is what makes a changing band count safe.
        self._trace_idx = {t.name: i for i, t in enumerate(self.fig.data)}
        self._total_name = f"fit_{self._id}_y0"
        expected = [self._total_name] + [f"y0_g{k}" for k in range(len(energies))]
        missing = [n for n in expected if n not in self._trace_idx]
        if missing:
            raise ValueError(
                f"check_plot did not produce the expected band traces; missing "
                f"{missing}. It returned {[t.name for t in self.fig.data]!r}."
            )
        self._apply_tie()
        self._redraw()

    # -- state ---------------------------------------------------------------

    @property
    def n_bands(self) -> int:
        return len(self.boxes) // 3

    @property
    def tied(self) -> bool:
        """Whether every band's width is one shared parameter."""
        return bool(self._tie_cb.value)

    def state(self):
        """Read the boxes back as ``(centers, widths, intens)``.

        Widths are per band even when tied -- they are all equal in that case, which
        matches how ``details`` reports a shared width once per band.
        """
        vals = [b.value for b in self.boxes]
        return vals[0::3], vals[1::3], vals[2::3]

    def _width_boxes(self):
        return self.boxes[1::3]

    def _fwhm_arg(self, widths):
        """What to hand fit_gaussians: a scalar when tied, else one width per band."""
        return widths[0] if self.tied else widths

    def _set_state(self, centers, widths, intens):
        """Write values into the boxes without triggering a redraw per box.

        Assigning 3N boxes would otherwise fire 3N observers; the caller redraws once.
        """
        ordered = []
        for i in range(len(centers)):
            ordered += [centers[i], widths[i], intens[i]]
        self._muted = True
        try:
            for box, value in zip(self.boxes, ordered):
                box.value = self._round(value)
        finally:
            self._muted = False

    def _apply_tie(self):
        """Mirror W0 into every width box and disable the rest when tied.

        Disabling rather than hiding keeps the row layout stable, and showing the shared
        value on every row matches what a tied fit reports back.
        """
        width_boxes = self._width_boxes()
        if not width_boxes:
            return
        tied = self.tied
        self._muted = True
        try:
            for i, box in enumerate(width_boxes):
                box.disabled = tied and i > 0
                if tied and i > 0:
                    box.value = width_boxes[0].value
        finally:
            self._muted = False

    # -- output plumbing -----------------------------------------------------

    @staticmethod
    def _render(out, figure):
        """Replace an Output widget's contents with `figure`.

        Builds the display bundle and *assigns* it, rather than displaying into the
        widget as a context manager. That is the only approach with no append path at
        all: `out.outputs` is a tuple, and assigning a one-element tuple leaves exactly
        one output however many times it is called.

        The alternatives all failed. `with out: clear_output(wait=True); display(fig)`
        appended a copy per redraw, so N redraws showed N stacked plots; plain
        `clear_output()` did too; and `out.outputs = ()` before the context manager did
        not help either, because the figure is not reliably captured into the widget in
        the first place. `append_display_data()` stores an output with no mime data, so
        a figure added that way renders as nothing.

        `pio.to_json` rather than `figure.to_plotly_json()`: it encodes numpy arrays in
        the binary form the renderer expects, where the raw dict would carry ndarrays
        that cannot cross the comm.
        """
        import json

        import plotly.io as pio

        out.outputs = (
            {
                "output_type": "display_data",
                "data": {
                    "application/vnd.plotly.v1+json": json.loads(pio.to_json(figure))
                },
                "metadata": {},
            },
        )

    @staticmethod
    def _log(out, *lines):
        """Replace an Output widget's contents with text, assigned for the same reason."""
        text = "\n".join(str(line) for line in lines)
        out.outputs = (
            {"output_type": "stream", "name": "stdout", "text": text + "\n"},
        )

    # -- handlers ------------------------------------------------------------

    def _redraw(self, *_):
        """Recompute the band curves and residuals, then re-render."""
        self.redraws += 1
        centers, widths, intens = self.state()
        total, components = band_curves(self._x, centers, widths, intens)

        self.fig.data[self._trace_idx[self._total_name]].y = total
        for k, comp in enumerate(components):
            self.fig.data[self._trace_idx[f"y0_g{k}"]].y = comp
        self._render(self._plot_out, self.fig)

        if self.resid_fig is not None:
            self.resid_fig.data[0].y = self._y - total
            self._render(self._resid_out, self.resid_fig)

    def _on_param_change(self, change):
        if self._muted:
            return
        # With widths tied, editing any width box propagates to the others before the
        # single redraw, so the display never shows a mixed state.
        if self.tied and change["owner"] in self._width_boxes():
            self._muted = True
            try:
                for box in self._width_boxes():
                    box.value = change["new"]
            finally:
                self._muted = False
        self._redraw()

    def _on_tie_change(self, _change):
        self._apply_tie()
        self._redraw()

    def _on_fit_click(self, _button):

        centers, widths, intens = self.state()
        self._pre_fit = (centers, widths, intens)  # so Revert can undo this
        results, details, fit = self._data.fit_gaussians(
            energies=centers,
            fwhm=self._fwhm_arg(widths),
            intens=intens,
            id=self._id,
            x_col=self._x_col,
            y_cols=self._y_col,
            xrange=self._xrange,
            verbose=0,
        )
        # results is always the expanded layout, one width per band, tied or not
        n = len(centers)
        self._set_state(results[:n], results[n : 2 * n], results[2 * n : 3 * n])
        self._redraw()
        self._log(self._log_out, details.to_markdown(), "", fit_quality(fit))

    def _on_revert_click(self, _button):
        """Restore the box values from just before the last fit."""

        if self._pre_fit is None:
            self._log(
                self._log_out,
                "Nothing to revert -- no fit has been run since the last revert.",
            )
            return
        self._set_state(*self._pre_fit)
        self._pre_fit = None
        self._redraw()
        self._log(self._log_out, "Reverted to the values from before the last fit.")

    def _on_add_band(self, _button):
        """Add a band where the current fit is worst.

        The new band is centered on the largest |residual|, takes that residual's signed
        value as its starting intensity, and copies the last band's width. That is both
        deterministic and usually close to where a band is actually needed -- centering
        on the middle of the window instead puts it at 1075 nm on a 280-1870 nm scan,
        which is nowhere near anything and has to be retyped every time.
        """
        centers, widths, intens = self.state()
        total, _ = band_curves(self._x, centers, widths, intens)
        residual = self._y - total
        worst = int(np.argmax(np.abs(residual)))
        centers = centers + [self._round(self._x[worst])]
        widths = widths + [widths[-1]]
        intens = intens + [self._round(residual[worst])]
        self._pre_fit = None  # the stashed values no longer match the band count
        self._rebuild(centers, widths, intens)

    def _on_remove_band(self, _button):

        centers, widths, intens = self.state()
        if len(centers) <= 1:
            self._log(self._log_out, "Cannot remove the last band.")
            return
        self._pre_fit = None
        self._rebuild(centers[:-1], widths[:-1], intens[:-1])

    def _on_code_click(self, _button):

        centers, widths, intens = self.state()
        self._log(
            self._log_out,
            fit_call_source(
                self._id,
                centers,
                self._fwhm_arg(widths),
                intens,
                x_col=self._x_col,
                y_cols=self._y_col,
                xrange=self._xrange,
                var_name=self._var_name,
                precision=self._precision,
            ),
        )

    # -- display -------------------------------------------------------------

    def _ipython_display_(self):
        from IPython.display import display

        display(self.view)


# ---------------------------------------------------------------------------
# Written by RG (Robert Gipson). Created by Claude (Opus 5): the interactive band
# fitting dashboard. fit_call_source and fit_quality are pure and unit-tested; the
# widget layer is a thin shell over band_curves, _pack_fitvars and
# fit_gaussians(verbose=0) from abscd.py. ipywidgets is imported lazily via
# _require_ipywidgets so `import sollabdata` does not need it. Rendering deliberately
# uses Output + a plain plotly Figure rather than FigureWidget, which in plotly 7.0.0
# raises KeyError: 'uid' from _handler_js2py_traceDeltas and stays broken across a
# kernel restart. Band traces are found by name (fit_<id>_y0, y0_g<k>) rather than by
# position, which is what lets bands be added and removed at runtime. The residual panel
# is a separate figure rather than another check_plot subplot row, so check_plot needs no
# knowledge of the dashboard. Output widgets are updated by ASSIGNING a one-element
# `outputs` tuple holding the display bundle, which is the only approach with no append
# path: clear_output(wait=True), plain clear_output() and clearing before a context
# manager all left redraws stacking a new figure each time. Tying the FWHM mirrors W0
# into the other width boxes,
# disables them, and passes a scalar fwhm, matching fit_gaussians' own convention.
# ---------------------------------------------------------------------------
