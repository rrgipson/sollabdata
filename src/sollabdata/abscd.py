"""Absorption / circular dichroism (Abs-CD) data from the J-1700.

Child class of :class:`~sollabdata.labdata.LabData` for spectra collected on the
J-1700. The base class handles info_df bookkeeping, copying, dropping, plotting,
and writing; this module adds J-1700 file parsing, reference subtraction, unit
conversions, and Gaussian band fitting.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.optimize import least_squares

from .labdata import LabData

# Some global variable
NM = "NANOMETERS"
WAVE = "Wavenums"
ABS = "ABSORBANCE"
CD = "CD/DC [mdeg]"

# np.trapz was renamed np.trapezoid in numpy 2.0 and the old name now warns/errors
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# The J-1700 labels its channels JCAMP-style: XUNITS for the x axis, then YUNITS,
# Y2UNITS, Y3UNITS, ... for the y channels in column order -- hence the optional
# digits. Anchored to the start of the line, because the older `"UNITS" in line and
# "X" in line` test misread any y channel whose unit string happened to contain an
# "X" (YUNITS,FLUX) as the x-axis label. Shared with VTVH_MCD.load in mcd.py, which
# parses the same header.
X_UNITS_RE = re.compile(r"^X\d*UNITS\s*,")
Y_UNITS_RE = re.compile(r"^Y\d*UNITS\s*,")


class AbsCD(LabData):
    """Abs/CD Data Class for data from the J-1700 with parent LabData.

    One info_df row per scan; the nested ``data`` frame holds the spectrum.
    """

    def __init__(
        self,
        experiment_df: pd.DataFrame | None = None,
        info_csv: str | None = None,
        info_df: pd.DataFrame | None = None,
        processing_metadata=None,
        path_to_raw_data: str | None = None,
        labels: dict | None = None,
        **kwargs,
    ) -> None:
        """See LabData.__init__ for the shared arguments; kwargs reach load() (e.g. j1700, verbose)."""
        super().__init__(
            experiment_df,
            info_csv,
            info_df,
            processing_metadata,
            path_to_raw_data,
            labels,
            **kwargs,
        )
        # drop empty columns -- check self.info_df, since process() may have just built it from info_csv
        if self.info_df is not None:
            self.info_df.drop(
                self.info_df.columns[
                    self.info_df.columns.str.contains("unnamed", case=False)
                ],
                axis=1,
                inplace=True,
            )

    def load(self, path_to_raw_data, data_files, j1700=True, verbose=False, **kwargs):
        """Convert from raw csv files to standardized (specifically for J-1700 Abs/CD).
        If data was saved using the code and not directly by the J-1700 set j1700=False to use parent function.
        """
        if not j1700:
            # Use parent load function if saved data from code rather than j1700
            print("Using Parent Class Load function.")
            return super().load(path_to_raw_data, data_files, **kwargs)

        # use pandas to read in and aggregate data files
        # parse data from each J-1700 data file
        # NOTE: x/y labels accumulate across every file in the folder, so all files in
        # one folder are assumed to share the same channel layout.
        xlabels = []
        ylabels = []
        # go through the data files
        for i, row in self.info_df.iterrows():
            if self.info_df.at[i, "File"] in data_files:
                if verbose:
                    print(self.info_df.at[i, "File"])
                file_path = os.path.join(path_to_raw_data, self.info_df.at[i, "File"])
                # open and read each file to extract necessary infomation for reading with pandas
                # npts resets per file so a missing NPOINTS line cannot silently reuse the previous file's count
                npts = None
                with open(file_path, "r") as f:
                    count = 1
                    for line in f:
                        if X_UNITS_RE.match(line):
                            xstr = line.split(",")[-1].replace("\n", "")
                            if xstr not in xlabels:
                                xlabels.append(xstr)
                        elif Y_UNITS_RE.match(line):
                            ystr = line.split(",")[-1].replace("\n", "")
                            if ystr not in ylabels:
                                ylabels.append(ystr)
                        elif "NPOINTS" in line:
                            npts = int(line.split(",")[-1])
                        elif "XYDATA" in line:
                            break
                        count = count + 1
                if npts is None:
                    raise ValueError(
                        f"No NPOINTS line found in {file_path}; is this a J-1700 export?"
                    )
                # read in csv as dataframe to correct place in info_df
                self.info_df.at[i, "data"] = pd.read_csv(
                    file_path,
                    skiprows=count,
                    header=None,
                    nrows=npts,
                    names=np.concatenate((xlabels, ylabels)),
                    dtype=float,
                    na_values=["1.#INF"],
                )
        return True

    def subtract(self, ref_id, ys, ids_to_subtract=None, drop_ref=False):
        """Subtract off 1 spectrum's y values from (by default) all of them.
        Include list of ids_to_subtract to only subtract ref_id off from a sub-set of the data.
        Does not check to make sure x values align!"""
        ref_idx = self.info_df.index[self.info_df["id"] == ref_id].to_list()
        if len(ref_idx) == 1:
            ref_idx = ref_idx[0]
        else:
            print("Did not select reference to subtract correctly.")
            return False
        # .copy() is defensive: the loop below overwrites the reference row's own data,
        # and this decouples the values we subtract from pandas' assignment semantics
        ref_values = self.info_df.at[ref_idx, "data"][ys].copy()
        for i, row in self.info_df.iterrows():
            if (
                ids_to_subtract is None or row["id"] in ids_to_subtract
            ):  # only subtract from selected samples
                self.info_df.at[i, "data"][ys] = row["data"][ys].sub(ref_values)
        # Drop the spectra used for the subtraction
        if drop_ref:
            self.drop(idx=ref_idx)
        # self.processing_metadata = self.processing_metadata + ' subtracted off ' + ref_id + ' from ' + str(ys)
        return True

    def add(self, ref_id, ys, ids_to_subtract=None, square=False):
        """Add 1 spectrum's y values to (by default) all of them.
        Include list of ids_to_subtract to only add ref_id to a sub-set of the data.
        square=True adds in quadrature instead, for propagating standard deviations.
        Does not check to make sure x values align!"""
        ref_idx = self.info_df.index[self.info_df["id"] == ref_id].to_list()
        if len(ref_idx) == 1:
            ref_idx = ref_idx[0]
        else:
            print("Did not select reference to subtract correctly.")
            return False
        # .copy() defensively, as in subtract()
        ref_values = self.info_df.at[ref_idx, "data"][ys].copy()
        if square:
            ref_values = ref_values.pow(2)
        for i, row in self.info_df.iterrows():
            if (
                ids_to_subtract is None or row["id"] in ids_to_subtract
            ):  # only subtract from selected samples
                if square:
                    self.info_df.at[i, "data"][ys] = (
                        row["data"][ys].pow(2).add(ref_values).pow(0.5)
                    )
                else:
                    self.info_df.at[i, "data"][ys] = row["data"][ys].add(ref_values)

        # self.processing_metadata = self.processing_metadata + ' added to ' + ref_id + ' from ' + str(ys)
        return True

    def baseline(self, ys, x_range, x_col="NANOMETERS", ignore_idx=None):
        """Subtract off average y value of featureless region to correct for non-zero baseline"""
        for i, row in self.info_df.iterrows():
            if ignore_idx is None or i not in ignore_idx:
                ref_values = (
                    row["data"]
                    .loc[
                        (row["data"][x_col] > x_range[0])
                        & (row["data"][x_col] < x_range[1])
                    ]
                    .mean(axis=0)[ys]
                )
                self.info_df.at[i, "data"][ys] = row["data"][ys].sub(ref_values)
        # self.processing_metadata = self.processing_metadata + ' baseline subtracted off using' + x_col + str(x_range)
        return True

    def fix_changeover(
        self, ys, x_change, x_col="NANOMETERS", to_move="Less", ignore_idx=None, pts=2
    ):
        """Fix the discontinuity created by the J-1700 when the detector changes"""
        for i, row in self.info_df.iterrows():
            if ignore_idx is None or i not in ignore_idx:
                # Determine the y value before and after the discontinuity
                before = np.average(
                    self.info_df.at[i, "data"]
                    .loc[self.info_df.at[i, "data"][x_col] < x_change]
                    .head(pts)[ys]
                    .values,
                    axis=0,
                )
                after = np.average(
                    self.info_df.at[i, "data"]
                    .loc[self.info_df.at[i, "data"][x_col] >= x_change]
                    .tail(pts)[ys]
                    .values,
                    axis=0,
                )
                diff = np.subtract(after, before)
                # subtract off the difference from the appropriate side
                if to_move == "Less" or to_move == "less":
                    self.info_df.at[i, "data"][ys] = pd.concat(
                        (
                            self.info_df.at[i, "data"]
                            .loc[self.info_df.at[i, "data"][x_col] < x_change][ys]
                            .add(diff),
                            self.info_df.at[i, "data"].loc[
                                self.info_df.at[i, "data"][x_col] >= x_change
                            ][ys],
                        )
                    )

                elif to_move == "Both" or to_move == "both":
                    self.info_df.at[i, "data"][ys] = pd.concat(
                        (
                            self.info_df.at[i, "data"]
                            .loc[self.info_df.at[i, "data"][x_col] < x_change][ys]
                            .add(diff / 2),
                            self.info_df.at[i, "data"]
                            .loc[self.info_df.at[i, "data"][x_col] >= x_change][ys]
                            .sub(diff / 2),
                        )
                    )
                elif to_move == "More" or to_move == "more":
                    self.info_df.at[i, "data"][ys] = pd.concat(
                        (
                            self.info_df.at[i, "data"].loc[
                                self.info_df.at[i, "data"][x_col] < x_change
                            ][ys],
                            self.info_df.at[i, "data"]
                            .loc[self.info_df.at[i, "data"][x_col] >= x_change][ys]
                            .sub(diff),
                        )
                    )
                else:
                    # bail out instead of falling through to the "Shifted" message below
                    print(
                        "Please specify to_move as Less, Both, or More as to_move parameter."
                    )
                    return False
                print(f'Shifted {row["id"]} {ys} by {diff}.')
        return True

    def add_eV(self, nm_str="NANOMETERS"):
        """Add an x value of wavenumbers by converting nanometers"""
        if nm_str in self.info_df["data"].iloc[0].columns:
            for i, row in self.info_df.iterrows():
                self.info_df.at[i, "data"]["eV"] = np.divide(
                    1239.8, row["data"][nm_str].astype(float)
                )
            print(self.info_df["data"].iloc[0].columns.values)
            return True
        else:
            return False

    def add_wavenums(self, nm_str="NANOMETERS"):
        """Add an x value of wavenumbers by converting nanometers"""
        if nm_str in self.info_df["data"].iloc[0].columns:
            for i, row in self.info_df.iterrows():
                self.info_df.at[i, "data"]["Wavenums"] = (
                    np.power(row["data"][nm_str].astype(float), -1) * 10000000
                )
            print(self.info_df["data"].iloc[0].columns.values)
            return True
        else:
            return False

    @staticmethod
    def _conc_to_M(conc, conc_units):
        """Convert a concentration array to molar. Raises on an unrecognized unit rather
        than leaving the converted value undefined."""
        factors = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9}
        if conc_units not in factors:
            raise ValueError(
                f"conc_units must be one of {sorted(factors)}, got {conc_units!r}."
            )
        return conc * factors[conc_units]

    def add_deps(self, conc, conc_units="M", path_length=1):
        """Add a y value of Delta Epsilon (1/M*cm) by converting from mdeg.
        conc - takes numerical value or name of dataframe column in self.data"""
        print("Path Length = ", path_length)

        # Handle concentration input type
        if isinstance(conc, str):
            conc = self.info_df[conc].to_numpy()
        else:
            try:
                conc = np.ones(len(self.info_df)) * float(conc)
            except (TypeError, ValueError):
                print("Please give conc as a number or df column name.")
                return False

        concM = self._conc_to_M(conc, conc_units)

        # check for units to convert from
        if CD in self.info_df["data"].iloc[0].columns:
            # for each row do the math for the conversion
            # enumerate gives the positional index into concM; i is the info_df label,
            # which is not the same thing once rows have been dropped without a reindex
            for pos, (i, row) in enumerate(self.info_df.iterrows()):
                self.info_df.at[i, "data"]["deps"] = np.divide(
                    row["data"][CD], (concM[pos] * path_length * 32980)
                )
        print(self.info_df["data"].iloc[0].columns.values)
        return True

    def add_eps(
        self, conc, conc_units="M", path_length=1, abs_str="ABSORBANCE", ids=None
    ):
        """Add a y value of Epsilon (1/M*cm) by converting from Abs.
        conc - takes numerical value or name of dataframe column in self.data
        ids - list of ids for which rows to add eps for"""
        # Handle concentration input type
        if isinstance(conc, str):
            conc = self.info_df[conc].to_numpy()
        else:
            try:
                conc = np.ones(len(self.info_df)) * float(conc)
            except (TypeError, ValueError):
                print("Please give conc as a number or df column name.")
                return False

        concM = self._conc_to_M(conc, conc_units)

        # check for units to convert from
        if abs_str in self.info_df["data"].iloc[0].columns:
            # for each row do the math for the conversion
            # enumerate gives the positional index into concM (see add_deps)
            for pos, (i, row) in enumerate(self.info_df.iterrows()):
                if ids is None or row["id"] in ids:
                    self.info_df.at[i, "data"]["eps"] = np.divide(
                        row["data"][abs_str], (concM[pos] * path_length)
                    )
        print(self.info_df["data"].iloc[0].columns.values)
        return True

    def gauss(self, x, center, fwhm):
        """Define the Gaussian Distribution Function"""
        width = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return np.exp(-1 / 2 * (x - center) ** 2 / (width) ** 2)

    def resid(self, fitvars, xs, ys):
        """Residual Calculator for Fitting to Gaussians"""
        # fit vars should be in format [energy1, energy2, ..., width1, w2, ..., scalarAbs1, sA2, ..., scalarCD1, sCD2, ...]
        num_gauss = int(len(fitvars) / (len(ys) + 2))

        total_resid = np.array([])
        # for each y given, calculate gaussians, fit, and add to list of residuals
        for j in range(len(ys)):
            # check to see if multiple x lists
            if len(xs) == len(ys):
                x = xs[j]
            else:
                x = xs
            # get a list of the individual gaussian y values
            gauss_list = [
                fitvars[i + ((2 + j) * num_gauss)]
                * self.gauss(x, fitvars[i], fitvars[i + num_gauss])
                for i in range(num_gauss)
            ]
            # calculate total for Abs with current params
            total_fit = np.sum(gauss_list, axis=0)
            # calculate total residual and add to the list
            total_resid = np.concatenate((total_resid, ys[j] - total_fit))

        return total_resid

    def fit_gaussians(
        self,
        energies,
        fwhm,
        intens,
        id,
        x_col=WAVE,
        y_cols=("eps", "deps"),
        xrange=None,
        low_bds=None,
        up_bds=None,
        same_x=True,
        gtol=1e-13,
        ftol=1e-13,
        xtol=1e-13,
        scalar=None,
    ):
        """A function to fit Abs/CD data to gaussian bands.
        y_cols defaults to a tuple rather than a list so the default cannot be mutated by a caller.
        """
        # make sure all guess inputs are floats
        energies = [float(e) for e in energies]
        widths = [float(w) for w in fwhm]
        # intens = np.concatenate((intens))
        intens = [float(ints) for ints in intens]

        num_gauss = len(energies)
        # get x and y data
        idx = self.info_df.index[self.info_df["id"] == id].to_numpy()[0]
        sample_row = self.info_df.loc[self.info_df["id"] == id].copy()
        sample_row = sample_row.reset_index()
        if xrange is not None:
            row_data = sample_row.at[0, "data"].loc[
                (sample_row.at[0, "data"][x_col] > xrange[0])
                & (sample_row.at[0, "data"][x_col] < xrange[1])
            ]
        else:
            row_data = sample_row.at[0, "data"]
        xs = row_data[x_col].to_numpy()
        # a bare string means one y column; any other sequence means several
        # (checking `is not list` here would mis-handle a tuple or an Index)
        ys = (
            [row_data[y_cols].to_numpy()]
            if isinstance(y_cols, str)
            else row_data[list(y_cols)].T.to_numpy()
        )

        # normalize ys, intensities, and bounds
        # float dtype: an int array would truncate each area and normalize by the wrong value
        # (and divide by zero for any area below 1)
        areas = np.zeros(len(ys), dtype=float)
        nys = []
        nintens = np.array(intens)
        # handle bounds setup
        if low_bds is not None:
            low_bds = [float(lb) for lb in low_bds]
            nl_bds = np.array(low_bds)
        else:
            nl_bds = -1 * np.inf
        if up_bds is not None:
            up_bds = [float(ub) for ub in up_bds]
            nu_bds = np.array(up_bds)
        else:
            nu_bds = np.inf
        # iterate through ys, calc and store areas, and normalize
        for k in range(len(ys)):
            # abs() on the integral, not just on ys: the J-1700 writes x descending
            # (DELTAX,-1), so the integral comes back negative. The sign cancels out
            # for the data and intensities, but dividing low_bds/up_bds by a negative
            # area swaps them, and least_squares then rejects the bounds outright.
            # The area is only ever used as a magnitude scale factor.
            areas[k] = abs(_trapezoid(abs(ys[k]), x=xs if same_x else xs[k]))
            if scalar is not None:
                areas[k] = areas[k] / scalar[k]
            nys.append(np.divide(ys[k], areas[k]))
            nintens[k * num_gauss : k * num_gauss + num_gauss] = np.divide(
                intens[k * num_gauss : k * num_gauss + num_gauss], areas[k]
            )

            # normalize bounds
            if low_bds is not None:
                nl_bds[
                    k * num_gauss + (2 * num_gauss) : k * num_gauss + (3 * num_gauss)
                ] = np.divide(
                    low_bds[
                        (2 * num_gauss)
                        + k * num_gauss : k * num_gauss
                        + (3 * num_gauss)
                    ],
                    areas[k],
                )
            if up_bds is not None:
                nu_bds[
                    k * num_gauss + (2 * num_gauss) : k * num_gauss + (3 * num_gauss)
                ] = np.divide(
                    up_bds[
                        (2 * num_gauss)
                        + k * num_gauss : k * num_gauss
                        + (3 * num_gauss)
                    ],
                    areas[k],
                )

        # prepare input lists
        params = np.concatenate((energies, widths, nintens))
        bounds = (nl_bds, nu_bds)
        # run fit
        fit = least_squares(
            self.resid,
            params,
            bounds=bounds,
            args=(xs, nys),
            verbose=1,
            gtol=gtol,
            ftol=ftol,
            xtol=xtol,
        )

        # print(fit['success'])
        # print(fit['message'])
        nresults = fit["x"]

        results = np.array(nresults)
        for ai in range(len(areas)):
            results[
                ai * num_gauss + (2 * num_gauss) : ai * num_gauss + (3 * num_gauss)
            ] = np.multiply(
                nresults[
                    ai * num_gauss + (2 * num_gauss) : ai * num_gauss + (3 * num_gauss)
                ],
                areas[ai],
            )

        details = pd.DataFrame()
        details["Energy"] = results[0:num_gauss]
        details["FWHM"] = results[num_gauss : 2 * num_gauss]
        for i in range(len(ys)):
            ylab = "Inten_y" + str(i)
            details[ylab] = results[(i + 2) * num_gauss : (i + 3) * num_gauss]
        # add Abs max value
        # details['y0_max'] = np.multiply([gauss(0,0,w) for w in details['FWHM']],details['Inten_y0'])
        # Calc fwhm and oscillator strengths (f)
        fs = 4.61e-9 * details["FWHM"] * details["Inten_y0"]
        # details['fwhm'] = fwhms
        details["f"] = fs

        # Save results to data object
        if "fit" not in self.info_df.columns:
            self.info_df["fit"] = None
            self.info_df["fit"] = self.info_df["fit"].astype("object")
        print(f"Fit params stored in info_df.at[{idx}, 'fit']")
        print(details.to_markdown())
        self.info_df.at[idx, "fit"] = results

        return results, details, fit

    def check_plot(
        self,
        id: str,
        result=None,
        x_col=WAVE,
        y_cols=("eps", "deps"),
        xrange=None,
        *args,
        **kwargs,
    ):
        """A way to plot the fits performed by fit_gaussians
        id specifies which one to use
        specify result if don't want to use the result automatically stored in row['fit']
        **kwargs is passed through to quick_plot"""
        if result is None:
            idx = self.info_df.index[self.info_df["id"] == id].to_numpy()[0]
            fitvars = self.info_df.at[idx, "fit"]
        else:
            fitvars = result

        # get x and y data
        sample_row = self.info_df.loc[self.info_df["id"] == id].copy()
        sample_row = sample_row.reset_index()
        if xrange is not None:
            # filter on the x column, not on the whole dataframe
            row_data = sample_row.at[0, "data"].loc[
                (sample_row.at[0, "data"][x_col] > xrange[0])
                & (sample_row.at[0, "data"][x_col] < xrange[1])
            ]
        else:
            row_data = sample_row.at[0, "data"]
        xs = row_data[x_col].to_numpy()
        ys = (
            [row_data[y_cols].to_numpy()]
            if isinstance(y_cols, str)
            else row_data[list(y_cols)].T.to_numpy()
        )

        # fit vars should be in format [energy1, energy2, ..., width1, w2, ..., scalarAbs1, sA2, ..., scalarCD1, sCD2, ...]
        num_gauss = int(len(fitvars) / (len(ys) + 2))

        # create dataframe for the results
        fit = pd.DataFrame()
        fit["x"] = xs
        # group the fit traces by which y they belong to, so each one can go onto the
        # matching quick_plot subplot row below
        traces_by_y = []
        # for each y given, calculate gaussians and collect them for plotting
        for j in range(len(ys)):
            ylab = "y" + str(j)
            x = xs

            # get a list of the individual gaussian y values
            gauss_list = [
                fitvars[i + ((2 + j) * num_gauss)]
                * self.gauss(x, fitvars[i], fitvars[i + num_gauss])
                for i in range(num_gauss)
            ]
            # calculate total for Abs with current params
            total_fit = np.sum(gauss_list, axis=0)

            # Add total fit, and each gaussian
            total_lab = "fit_" + id + "_" + ylab
            fit[total_lab] = total_fit.copy()
            row_traces = [(total_lab, total_fit)]
            for k in range(len(gauss_list)):
                lab = ylab + "_g" + str(k)
                fit[lab] = gauss_list[k]
                row_traces.append((lab, gauss_list[k]))
            traces_by_y.append(row_traces)

        # Plot the data FIRST so quick_plot builds the subplot grid (one row per y),
        # then drop the fit traces onto the matching rows. Handing quick_plot a flat
        # px.line figure instead raised "(row, col) pair sent is out of range" as soon
        # as more than one y column was fitted.
        if "ids" not in kwargs:
            kwargs["ids"] = [id]
        fig = self.quick_plot(y=y_cols, x=x_col, **kwargs)
        for j, row_traces in enumerate(traces_by_y):
            for lab, values in row_traces:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=values,
                        name=lab,
                        mode="lines",
                        line=dict(width=1, dash="dash"),
                    ),
                    row=j + 1,
                    col=1,
                )
        return fig
        # return fit


# ---------------------------------------------------------------------------
# Code written by RG (Robert Gipson).
# Claude (Opus 5) reviewed and adjusted this file:
#   - Added the missing `os` / `numpy` / `pandas` / `plotly.express` imports; the
#     module referenced pd, np and px without importing any of them, so it could
#     not be imported at all.
#   - Renamed the class to AbsCD (PEP 8, matches the module) and merged the
#     duplicated class docstring.
#   - Replaced the string parameter annotations on __init__ with real type hints.
#   - The "drop unnamed columns" step now checks `self.info_df`, so it also runs
#     when info_df came from info_csv via process().
#   - load(): os.path.join instead of string concatenation, resets npts per file
#     (it previously leaked the previous file's value), accepts **kwargs, and
#     returns True like the base class.
#   - subtract()/add(): copy the reference values defensively, since the loop
#     overwrites the reference row's own data as it goes.
#   - add_deps()/add_eps(): index concM positionally -- an info_df label was being
#     used as an array offset, which breaks after a drop without a reindex. Unit
#     conversion moved to a shared _conc_to_M that raises on an unknown unit
#     instead of leaving concM undefined; bare `except:` narrowed.
#   - Replaced `self.info_df.at[0,'data']` with `self.info_df['data'].iloc[0]` so
#     these still work on a non-zero-based index.
#   - fit_gaussians(): areas array is now float (an int array truncated every area
#     and divided by zero for areas below 1); mutable list defaults changed to
#     tuples; np.trapz aliased for numpy 2.0.
#   - check_plot(): filter on the x column rather than the whole dataframe; plot
#     every y's fit traces instead of only the last one's; and call quick_plot
#     first so the subplot grid exists before the fit traces are added -- handing
#     quick_plot a flat px.line figure raised "(row, col) pair sent is out of
#     range" whenever more than one y column was fitted.
#   - fix_changeover() returns instead of falling through to a success message.
#   - Channel labels are matched with anchored X_UNITS_RE / Y_UNITS_RE patterns
#     (XUNITS, YUNITS, Y2UNITS, ...) rather than `"UNITS" in line and "X" in line`,
#     which misread a y channel whose unit contained an "X" (e.g. FLUX) as the x
#     axis and shifted every column name by one. Shared with VTVH_MCD.load.
# ---------------------------------------------------------------------------
