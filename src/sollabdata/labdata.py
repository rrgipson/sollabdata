"""A set of classes and functions for reading in, storing, and analyzing Solomon Lab Data.

LabData serves as the parent class and each experiment and/or instrument type will
have a child class. The core data model is a single "info" DataFrame (``info_df``) in
which every row is one measurement: the flat columns hold metadata (id, file name,
scan number, ...) and the ``data`` column holds a nested per-measurement DataFrame of
the actual x/y traces.
"""

from __future__ import annotations

import os
import textwrap
from typing import Any, Callable, Iterable, Sequence

import matplotlib.pyplot as plt  # noqa: F401  (kept for interactive/child-class use)
import numpy as np  # noqa: F401  (kept for interactive/child-class use)
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

#: Default line colors for quick_plot, cycled per-trace. Defined at module level
#: (as an immutable tuple) so it is not a mutable default argument and so callers
#: can import and extend it. Alternative: plotly.express.colors.qualitative.G10
DEFAULT_COLORS: tuple[str, ...] = (
    "blue",
    "red",
    "green",
    "purple",
    "deepskyblue",
    "orange",
    "slategrey",
    "brown",
    "black",
    "gold",
)

#: File extensions treated as raw data files by LabData.read (compared lowercased).
DATA_EXTENSIONS: tuple[str, ...] = (".csv", ".txt")


def _as_list(value: Any) -> list:
    """Coerce a scalar (or None) into a list so ``Series.isin`` behaves as expected.

    Guards against the classic bug where a single string id is passed to ``isin``
    and gets iterated character by character.
    """
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, Iterable):
        return [value]
    return list(value)


# ---------------------------------------------------------------------------
# Parent Class
# ---------------------------------------------------------------------------
class LabData:
    """Parent class with a variety of general functions for processing Solomon Lab Data.

    Attributes:
        experiment_df: DataFrame with experiment_id, type, project (from the
            experiment dashboard).
        info_df: DataFrame with one row per measurement; holds the id and metadata
            columns (same as the info/key csv) plus a nested ``data`` column.
        processing_metadata: Record of how the data was processed. Format TBD
            (currently free-form, likely a str or dict).
        labels: Dict of abbreviations -> display names for ``info_df`` column headers.
    """

    def __init__(
        self,
        experiment_df: pd.DataFrame | None = None,
        info_csv: str | None = None,
        info_df: pd.DataFrame | None = None,
        processing_metadata: Any = None,
        path_to_raw_data: str | None = None,
        labels: dict | None = None,
        **kwargs,
    ) -> None:
        """Build a LabData object, optionally parsing raw data off disk.

        Args:
            experiment_df: See class Attributes.
            info_csv: File name of the experiment key. Resolved relative to
                ``path_to_raw_data`` if it is not a path that already exists.
                (Only used at construction time -- may be dropped as an argument later.)
            info_df: See class Attributes.
            processing_metadata: See class Attributes.
            path_to_raw_data: Path to the folder holding the info/data csv files.
                If given, ``process`` is called to populate ``info_df``.
            labels: See class Attributes.
            **kwargs: Forwarded to ``process`` -> ``load`` for instrument-specific
                loading options.
        """
        self.experiment_df: pd.DataFrame | None = experiment_df
        self.info_df: pd.DataFrame | None = info_df
        self.processing_metadata = processing_metadata
        self.labels = labels
        if path_to_raw_data is not None:
            print(f"Parsing Data from {path_to_raw_data} and {info_csv}")
            self.process(path_to_raw_data, info_csv, **kwargs)

    # -- object management --------------------------------------------------

    def copy(self) -> "LabData":
        """Copy an instance of the data Class. Should work for child classes too.

        Deep-copies the nested per-measurement DataFrames in the ``data`` column,
        which a plain ``info_df.copy()`` would leave shared between the two objects.
        """
        new = type(self)(
            experiment_df=(
                None if self.experiment_df is None else self.experiment_df.copy()
            ),
            info_df=None if self.info_df is None else self.info_df.copy(),
            processing_metadata=self.processing_metadata,
            labels=self.labels,
        )
        if new.info_df is not None and "data" in new.info_df.columns:
            for idx in new.info_df.index:
                cell = new.info_df.at[idx, "data"]
                if cell is not None and hasattr(cell, "copy"):
                    new.info_df.at[idx, "data"] = cell.copy()
        return new

    def drop(
        self,
        ids: Sequence | str | None = None,
        idx: Sequence | int | None = None,
        reset_index: bool = True,
    ) -> None:
        """Drop a row of data from the info_df by either index (idx) or id column."""
        if ids is None and idx is None:
            raise ValueError("Pass either ids or idx to drop.")
        if ids is not None and idx is None:
            idx = self.get_idx_from_id(ids)
        self.info_df.drop(index=idx, inplace=True)
        if reset_index:
            self.info_df.reset_index(inplace=True, drop=True)
        return

    # -- reading / writing --------------------------------------------------

    def read(self, path_to_raw_data: str, info_csv: str | None) -> list[str]:
        """Given a folder with an info csv and data csvs, read the files, and fill in the info df without data.

        Returns:
            The list of data file names (not full paths) found in the folder.
        """
        # Resolve the info csv: accept either a bare file name in the raw data
        # folder or a full path that the caller already resolved.
        info_path = None
        if info_csv is not None:
            info_path = (
                info_csv
                if os.path.isfile(info_csv)
                else os.path.join(path_to_raw_data, info_csv)
            )
        info_name = None if info_csv is None else os.path.basename(info_csv)

        # Collect the available data files from the top level of the folder only
        # (os.scandir rather than os.walk, since subfolders are deliberately ignored).
        data_files = sorted(
            entry.name
            for entry in os.scandir(path_to_raw_data)
            if entry.is_file()
            and entry.name.lower().endswith(DATA_EXTENSIONS)
            and entry.name != info_name
        )

        # Use pandas to read in the info file, or synthesize a generic one
        if info_path is not None:
            self.info_df = pd.read_csv(info_path)
        elif self.info_df is None:
            # Generate a generic info df based on the files that were found
            self.info_df = pd.DataFrame(
                {
                    "id": data_files,
                    "ScanNum": range(1, len(data_files) + 1),
                    "File": data_files,
                }
            )

        # Create a column in info_df to hold the nested per-measurement data.
        # dtype=object so each cell can hold a whole DataFrame.
        self.info_df["data"] = pd.Series(dtype="object")
        return data_files

    def load(self, path_to_raw_data: str, data_files: Sequence[str], **kwargs) -> bool:
        """Convert data from raw csv files to standardized (will be instrument-specific) and put into self.info_df['data'].

        Child classes are expected to override this with instrument-specific parsing;
        ``**kwargs`` is accepted here so ``process`` can forward loader options.
        """
        # Iterate through the info_df to read in the data files and store in the
        # data column of info_df. os.path.join means path_to_raw_data does not
        # need a trailing separator.
        for idx in self.info_df.index:
            file_name = self.info_df.at[idx, "File"]
            if file_name in data_files:
                self.info_df.at[idx, "data"] = pd.read_csv(
                    os.path.join(path_to_raw_data, file_name), **kwargs
                )
        return True

    def process(self, path_to_raw_data: str, info_csv: str | None, **kwargs) -> bool:
        """Read and Load the data to populate the class attributes."""
        data_files = self.read(path_to_raw_data, info_csv)
        self.load(path_to_raw_data, data_files, **kwargs)
        return True

    def write(self, path_to_proc_data: str) -> str:
        """Write processed data to a folder, saving the info_df and the data separately as csv files.

        NOTE: file names are reused from the ``File`` column, so writing back into
        the raw data folder will overwrite the originals -- pass a new path.
        """
        # Fix path string if not given a terminal /
        if not path_to_proc_data.endswith("/"):
            path_to_proc_data = path_to_proc_data.strip().replace(" ", "_") + "/"
        # Check for and/or make the directory (makedirs handles nested paths)
        os.makedirs(path_to_proc_data, exist_ok=True)
        # Save each individual dataset
        for idx in self.info_df.index:
            data = self.info_df.at[idx, "data"]
            if data is None:
                continue
            # Manipulate the File name to make it a csv, and write the corrected
            # name back to info_df so the saved info.csv matches the files on disk.
            name = str(self.info_df.at[idx, "File"])
            if not name.lower().endswith(".csv"):
                name = name + ".csv"
            self.info_df.at[idx, "File"] = name
            # Save x,y data to csv
            data.to_csv(os.path.join(path_to_proc_data, name))
        # Save the info df (without the nested data column, which does not
        # serialize usefully to csv)
        info = self.info_df.drop("data", axis=1)
        info.to_csv(os.path.join(path_to_proc_data, "info.csv"), index=False)
        print(f"Data saved to {path_to_proc_data}.")
        return path_to_proc_data

    # -- display -----------------------------------------------------------

    def print(self) -> None:
        """Print a short summary of the data object."""
        print(repr(self))
        if self.info_df is not None:
            print(self.info_df.drop(columns="data", errors="ignore"))

    def __repr__(self) -> str:
        n_rows = 0 if self.info_df is None else len(self.info_df)
        return f"<{type(self).__name__}: {n_rows} rows>"

    def to_md(self) -> str:
        """Print some information and return the info dataframe in markdown for display using IPython."""
        print("Data Columns and info_df:")
        if len(self.info_df) and self.info_df["data"].iloc[0] is not None:
            # iloc[0] rather than at[0, ...] so this still works after a
            # drop(reset_index=False) has left a non-zero-based index
            print(self.info_df["data"].iloc[0].columns.values)
        return self.info_df.drop("data", axis=1).to_markdown()

    # -- plotting ----------------------------------------------------------

    def quick_plot(
        self,
        x: str | None = None,
        y: str | Sequence[str] | None = None,
        x_range: Sequence | None = None,
        y_range: Sequence | None = None,
        ids: Sequence | str | None = None,
        height: int | None = None,
        width: int = 1000,
        fig: go.Figure | None = None,
        colors: Sequence[str] = DEFAULT_COLORS,
        selection_criteria: Any = None,
        **kwargs,
    ) -> go.Figure:
        """Use plotly to generate a general plot: interactive subplots, one row per y column.

        Args:
            x: Column name in the nested data frames to use as x. Defaults to the
                first column.
            y: Column name (or list of names) to plot; one subplot row per entry.
                Defaults to every column after the first.
            x_range: [xmin, xmax] applied to all subplots.
            y_range: [ymin, ymax], or a list of such pairs (one per y) when
                multiple y columns are plotted.
            ids: Restrict the traces to these ids.
            height: Figure height; defaults to 350 per subplot row.
            width: Figure width.
            fig: Existing figure to add traces to, e.g. to overlay two objects.
            colors: Line colors, cycled across traces within each subplot.
            selection_criteria: Boolean mask / index selector applied to info_df
                (e.g. ``obj.info_df['type'] == 'CV'``) to pick a subset of rows.
            **kwargs: Forwarded to ``fig.update_layout`` (title, legend, ...).
        """
        # Select the rows to plot up front rather than mutating self.info_df,
        # so an error mid-plot cannot leave the object in a filtered state.
        plot_df = (
            self.info_df
            if selection_criteria is None
            else self.info_df.loc[selection_criteria]
        )
        if ids is not None:
            plot_df = plot_df.loc[plot_df["id"].isin(_as_list(ids))]
        if not len(plot_df):
            raise ValueError(
                "No rows left to plot after applying ids/selection_criteria."
            )

        # Set up y data structure, should be a list to handle plotting multiple y values stacked
        first_data = plot_df["data"].iloc[0]
        if y is None:
            y = list(first_data.columns.values[1:])
        if isinstance(y, str):
            y = [y]
        # Set up x data structure, should be the column name to use as x
        if x is None:
            x = first_data.columns.values[0]

        if fig is None:
            fig = plotly.subplots.make_subplots(
                rows=len(y), cols=1, vertical_spacing=0.1, shared_xaxes=True
            )  # subplot_titles=y,

        # One subplot row per y column; within a row, cycle colors across traces and
        # tie the traces for a given id together via legendgroup so clicking the
        # legend toggles that id in every subplot at once.
        for i, y_label in enumerate(y):
            for j, idx in enumerate(plot_df.index):
                color = colors[j % len(colors)]
                # Include both the id and the y label in the trace name
                trace_name = f'{plot_df.at[idx, "id"]} ({y_label})'
                fig.add_trace(
                    go.Scatter(
                        x=plot_df.at[idx, "data"][x],
                        y=plot_df.at[idx, "data"][y_label],
                        name=trace_name,
                        mode="lines",
                        line=dict(width=2, color=color),
                        legendgroup=str(plot_df.at[idx, "id"]),
                    ),
                    row=i + 1,
                    col=1,
                )
            fig.update_xaxes(title_text=x, row=i + 1, col=1)
            fig.update_yaxes(title_text=y_label, row=i + 1, col=1)
            fig.add_hline(y=0, row=i + 1, col=1)  # type: ignore[arg-type]

            # Setup default view ranges
            if y_range is not None:
                fig.update_yaxes(
                    range=y_range[i] if len(y) > 1 else y_range, row=i + 1, col=1
                )
            if x_range is not None:
                fig.update_xaxes(range=x_range, row=i + 1, col=1)

        # Update plot layout. Pass e.g. title=... or
        # legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01) through kwargs.
        if height is None:
            height = 350 * len(y) if len(y) > 1 else 400
        fig.update_layout(
            width=width, height=height, margin=dict(b=50, t=50, l=20), **kwargs
        )
        return fig

    def prep_plt(
        self, x: str | None = None, y: str | None = None, var_name: str = "to_plot"
    ) -> None:
        """Print the commands to generate a matplotlib plot of the data.

        The output is meant to be pasted into a notebook cell and then edited by
        hand, which is why it is emitted as source text rather than a figure.
        ``var_name`` is the name the object goes by in that notebook.
        """
        # Plot one line per row of info_df, then a block of axis-formatting calls
        # left commented/empty for hand-tuning.
        plot_lines = "\n".join(
            f"axs.plot({var_name}.info_df.at[{idx},'data']['{x}'], "
            f"{var_name}.info_df.at[{idx},'data']['{y}'], "
            f"label='{self.info_df.at[idx, 'id']}')"
            for idx in self.info_df.index
        )
        # dedent first, then substitute the (multi-line) plot calls in via replace
        # rather than str.format, so ids containing braces cannot break the template.
        template = textwrap.dedent(f"""\
            # {var_name} = VARNAME
            fig,axs = plt.subplots(figsize=[10,6])

            PLOT_LINES

            # axs.set_title()
            axs.set_ylabel('{y}')
            axs.set_xlabel('{x}')
            axs.legend()
            axs.set_xlim()
            axs.set_ylim()
            # axs.ticklabel_format(axis='x',style='sci',scilimits=(3,3),useMathText=True)

            plt.show()""")
        print(template.replace("PLOT_LINES", plot_lines))

    def prep_animation_plt(
        self, x: str | None = None, y: str | None = None, var_name: str = "to_plot"
    ) -> None:
        """Print the commands to generate a stepped/animated matplotlib plot of the data.

        Emits a loop that draws one figure per row, with all previously drawn
        traces faded behind the current one -- useful for stepping through a series.
        """
        print(textwrap.dedent(f"""\
                # {var_name} = VARNAME
                {var_name}.info_df['plot_label'] = {var_name}.info_df['id']
                # print({var_name}.info_df[['id', 'plot_label']])
                idx_list = {list(self.info_df.index)}
                x_col = '{x}'
                y_col = '{y}'

                if True:
                    for i, idx in enumerate(idx_list):
                        fig,axs = plt.subplots(figsize=[8,4])
                        # Faded traces for everything already stepped through
                        for j in range(i):
                            axs.plot({var_name}.info_df.at[idx_list[j],'data'][x_col], {var_name}.info_df.at[idx_list[j],'data'][y_col], label={var_name}.info_df.at[idx_list[j],'plot_label'], alpha=0.4)
                        axs.plot({var_name}.info_df.at[idx,'data'][x_col], {var_name}.info_df.at[idx,'data'][y_col], label={var_name}.info_df.at[idx,'plot_label'])

                        # axs.set_title()
                        axs.set_ylabel('{y}')
                        axs.set_xlabel('{x}')
                        axs.legend()
                        axs.set_xlim()
                        axs.set_ylim()
                        # axs.ticklabel_format(axis='x',style='sci',scilimits=(3,3),useMathText=True)

                        plt.show()"""))

    # -- row selection / batch operations ----------------------------------

    def get_idx_from_id(self, ids: Sequence | str) -> list:
        """Return the info_df index labels for the given id (or list of ids)."""
        return self.info_df.index[self.info_df["id"].isin(_as_list(ids))].to_list()

    def apply_by_row(
        self,
        func: Callable,
        col_name: str | None = None,
        ids: Sequence | str | None = None,
        expand: bool = False,
        prefix: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Batch-run a function that takes a row's id as its first argument (e.g. a bound method like self.get_thermo)
        over every row (or the subset given by ids), collecting the results into info_df.
        By default the raw result is stored as an object in a single new column named col_name.
        If expand=True, func is expected to return a dict, and each key is instead written to its own
        info_df column (named `key`, or `{prefix}_{key}` if prefix is given) rather than nested in col_name.
        """
        idxs = self.info_df.index if ids is None else self.get_idx_from_id(ids)
        if not expand:
            if col_name is None:
                raise ValueError("col_name is required when expand=False.")
            if col_name not in self.info_df.columns:
                self.info_df[col_name] = pd.Series(dtype="object")
            for idx in idxs:
                self.info_df.at[idx, col_name] = func(
                    self.info_df.at[idx, "id"], **kwargs
                )
        else:
            for idx in idxs:
                result = func(self.info_df.at[idx, "id"], **kwargs)
                for key, value in result.items():
                    col = key if prefix is None else f"{prefix}_{key}"
                    if col not in self.info_df.columns:
                        self.info_df[col] = pd.Series(dtype="object")
                    self.info_df.at[idx, col] = value
        return self.info_df


#: Backwards-compatible alias for the old, non-PEP-8 class name. Safe to delete
#: once no notebooks reference `Lab_Data`.
Lab_Data = LabData


# ---------------------------------------------------------------------------
# Code written by RG (Robert Gipson).
# Claude (Opus 5) made the following additions/adjustments in this pass:
#   - Replaced the descriptive string annotations on __init__ with real type hints
#     (dropping the accompanying `# type: ignore`s) and moved those descriptions
#     into class/method docstrings.
#   - Fixed the `self.experimen_df` -> `self.experiment_df` typo.
#   - Moved the quick_plot `colors` mutable default list to a module-level
#     DEFAULT_COLORS tuple; added a DATA_EXTENSIONS constant and an `_as_list`
#     helper so a single string id no longer gets iterated character-by-character.
#   - quick_plot now filters into a local `plot_df` instead of temporarily
#     swapping out self.info_df, and hoists the id filter out of the inner loop.
#   - Path handling: os.path.join throughout (no trailing-separator requirement),
#     info_csv resolved relative to path_to_raw_data when it isn't already a path,
#     os.makedirs(exist_ok=True) instead of os.mkdir, os.scandir instead of
#     os.walk, and case-insensitive extension matching.
#   - Bug fixes: `load` accepts **kwargs so `process` can forward them; `write`
#     writes the corrected .csv file name back into info_df instead of mutating
#     the throwaway iterrows() row; `copy` and `to_md` tolerate a None info_df /
#     non-zero-based index; `drop` raises when given neither ids nor idx.
#   - prep_plt / prep_animation_plt rewritten as dedented template strings, and
#     `print()` / `__repr__` filled in with a short summary.
#   - Added section comments and inline comments explaining the info_df data model.
#   - Renamed the class Lab_Data -> LabData (PEP 8) and kept a `Lab_Data` alias.
# ---------------------------------------------------------------------------
