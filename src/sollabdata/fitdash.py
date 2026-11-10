"""Interactive Gaussian band fitting for notebooks.

A small dashboard for the step that actually costs time in band fitting: landing on a
good initial guess. Numeric boxes hold each band's center, width and intensity; the
model curve redraws over the data whenever a value is committed; a Fit button runs the
real :meth:`~sollabdata.abscd.AbsCD.fit_gaussians` from the current values and snaps the
boxes to the answer; and a Show-call button emits the equivalent ``fit_gaussians(...)``
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
    fwhm: Sequence[float],
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

    Every value is interpolated with ``!r``, which is what keeps a column name like
    ``"CD/DC [mdeg]"`` -- or an id containing a quote or bracket -- from producing
    broken source.
    """

    def nums(values):
        return [round(float(v), precision) for v in values]

    lines = [
        f"# {var_name} = YOUR_OBJECT",
        f"results, details, fit = {var_name}.fit_gaussians(",
        f"    energies={nums(energies)!r},",
        f"    fwhm={nums(fwhm)!r},",
        f"    intens={nums(intens)!r},",
        f"    id={id!r},",
        f"    x_col={x_col!r},",
        f"    y_cols={y_cols!r},",
    ]
    if xrange is not None:
        lines.append(f"    xrange={list(xrange)!r},")
    lines.append(")")
    return "\n".join(lines)


class BandFitDashboard:
    """Boxes + live overlay + Fit button for one spectrum's Gaussian bands.

    Prefer :meth:`sollabdata.abscd.AbsCD.fit_dashboard`, which constructs this.

    Scope is deliberately one id and one y column. Multiple ids or y columns are
    supported by ``fit_gaussians`` itself and are better specified in pasted source
    than in a UI.

    Attributes:
        fig: the plotly Figure being redrawn. Its traces are
            ``[data, total_fit, g0, ... g(N-1)]``.
        boxes: the FloatText widgets, laid out ``[E0, W0, I0, E1, W1, I1, ...]``.
        redraws: how many redraws have happened, which makes the mute flag's effect
            observable from a test.

    Args to __init__ beyond the obvious:
        box_precision: decimal places shown in the boxes, and used for the emitted
            source. Values are rounded on the way in, so ``state()`` returns what is
            displayed rather than a longer float the box is truncating for show.
        box_width: CSS width of each box. The default fits ~10 digits while keeping
            three boxes on one line without horizontal scrolling; widen it if a band
            parameter needs more.
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
        box_precision: int = 3,
        box_width: str = "130px",
        **plot_kwargs,
    ) -> None:
        widgets = _require_ipywidgets()
        self._w = widgets

        energies = [float(v) for v in energies]
        fwhm = [float(v) for v in fwhm]
        intens = [float(v) for v in intens]
        n = len(energies)
        if not (len(fwhm) == len(intens) == n):
            raise ValueError(
                f"energies, fwhm and intens must be the same length, got "
                f"{n}, {len(fwhm)} and {len(intens)}."
            )
        if n == 0:
            raise ValueError("Pass at least one band.")

        self._data = data
        self._id = id
        self._x_col = x_col
        self._y_col = y_col
        self._xrange = xrange
        self._var_name = var_name
        self._precision = box_precision
        self._box_width = box_width
        self._n = n
        self._muted = False
        self.redraws = 0

        # The x grid the overlay is computed on, matching check_plot's own windowing.
        row = data.info_df.loc[data.info_df["id"] == id]
        if not len(row):
            raise ValueError(f"No info_df row with id {id!r}.")
        frame = row["data"].iloc[0]
        if xrange is not None:
            frame = frame.loc[(frame[x_col] > xrange[0]) & (frame[x_col] < xrange[1])]
        self._x = frame[x_col].to_numpy()

        # Build the figure ONCE, then mutate its band traces per redraw. Rebuilding via
        # check_plot every time costs ~10 ms and resets the axis ranges; reusing one
        # figure keeps the view stable across updates.
        self.fig = data.check_plot(
            id,
            result=data._pack_fitvars(energies, fwhm, [intens]),
            x_col=x_col,
            y_cols=y_col,
            xrange=xrange,
            **plot_kwargs,
        )
        # check_plot appends [total_fit, g0..g(N-1)] after the data trace(s)
        self._fit_start = len(self.fig.data) - (n + 1)
        if self._fit_start < 1:
            raise ValueError(
                f"check_plot returned {len(self.fig.data)} traces for {n} bands; "
                "expected the data trace(s) plus a total and one per band."
            )

        self.boxes, rows = self._make_boxes(energies, fwhm, intens)
        for box in self.boxes:
            box.observe(self._on_param_change, names="value")

        self._plot_out = widgets.Output()
        self._log_out = widgets.Output()
        fit_btn = widgets.Button(description="Fit", button_style="primary")
        code_btn = widgets.Button(description="Show call")
        fit_btn.on_click(self._on_fit_click)
        code_btn.on_click(self._on_code_click)

        self.view = widgets.VBox(
            [*rows, widgets.HBox([fit_btn, code_btn]), self._plot_out, self._log_out]
        )
        self._redraw()

    # -- construction helpers ----------------------------------------------

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
                    value=self._round(c), description=f"E{i}", step=10.0,
                    layout=layout, style=style,
                ),
                widgets.FloatText(
                    value=self._round(w), description=f"W{i}", step=10.0,
                    layout=layout, style=style,
                ),
                widgets.FloatText(
                    value=self._round(v), description=f"I{i}", step=0.1,
                    layout=layout, style=style,
                ),
            ]
            boxes += trio
            rows.append(widgets.HBox(trio))
        return boxes, rows

    def _round(self, value) -> float:
        """Round to the configured precision, so the boxes do not show noise digits."""
        return round(float(value), self._precision)

    # -- state ---------------------------------------------------------------

    def state(self):
        """Read the boxes back as ``(centers, widths, intens)``."""
        vals = [b.value for b in self.boxes]
        return vals[0::3], vals[1::3], vals[2::3]

    def _set_state(self, centers, widths, intens):
        """Write values into the boxes without triggering a redraw per box.

        Assigning 3N boxes would otherwise fire 3N observers; the caller redraws once.
        """
        ordered = []
        for i in range(self._n):
            ordered += [centers[i], widths[i], intens[i]]
        self._muted = True
        try:
            for box, value in zip(self.boxes, ordered):
                box.value = self._round(value)
        finally:
            self._muted = False

    # -- handlers ------------------------------------------------------------

    def _redraw(self, *_):
        """Recompute the band curves and re-render the figure."""
        from IPython.display import clear_output, display

        self.redraws += 1
        total, components = band_curves(self._x, *self.state())
        self.fig.data[self._fit_start].y = total
        for k, comp in enumerate(components):
            self.fig.data[self._fit_start + 1 + k].y = comp
        with self._plot_out:
            # wait=True holds the old render until the new one is ready, which is what
            # keeps the swap from flashing
            clear_output(wait=True)
            display(self.fig)

    def _on_param_change(self, _change):
        if self._muted:
            return
        self._redraw()

    def _on_fit_click(self, _button):
        from IPython.display import clear_output

        centers, widths, intens = self.state()
        results, details, _fit = self._data.fit_gaussians(
            energies=centers,
            fwhm=widths,
            intens=intens,
            id=self._id,
            x_col=self._x_col,
            y_cols=self._y_col,
            xrange=self._xrange,
            verbose=0,
        )
        n = self._n
        self._set_state(results[:n], results[n : 2 * n], results[2 * n : 3 * n])
        self._redraw()
        with self._log_out:
            clear_output(wait=True)
            print(details.to_markdown())

    def _on_code_click(self, _button):
        from IPython.display import clear_output

        centers, widths, intens = self.state()
        with self._log_out:
            clear_output(wait=True)
            print(
                fit_call_source(
                    self._id,
                    centers,
                    widths,
                    intens,
                    x_col=self._x_col,
                    y_cols=self._y_col,
                    xrange=self._xrange,
                    var_name=self._var_name,
                    precision=self._precision,
                )
            )

    # -- display -------------------------------------------------------------

    def _ipython_display_(self):
        from IPython.display import display

        display(self.view)


# ---------------------------------------------------------------------------
# Written by RG (Robert Gipson). Created by Claude (Opus 5): the interactive band
# fitting dashboard. fit_call_source is pure and unit-tested; the widget layer is a thin
# shell over band_curves, _pack_fitvars and fit_gaussians(verbose=0) from abscd.py.
# ipywidgets is imported lazily via _require_ipywidgets so `import sollabdata` does not
# need it. Rendering deliberately uses Output + a plain plotly Figure rather than
# FigureWidget, which in plotly 7.0.0 raises KeyError: 'uid' from
# _handler_js2py_traceDeltas and stays broken across a kernel restart.
# Box values are rounded to box_precision (3 dp) on the way in, and the boxes use a
# narrow width plus a narrow description label, because the ipywidgets defaults make a
# band's three boxes overflow the cell and force horizontal scrolling.
# ---------------------------------------------------------------------------
