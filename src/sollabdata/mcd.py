"""Magnetic circular dichroism (MCD) data.

MCD and VTVH-MCD spectra from the J-1700. MCD subclasses
:class:`~sollabdata.abscd.AbsCD` (which subclasses
:class:`~sollabdata.labdata.LabData`), reusing its J-1700 parsing and spectral
arithmetic and adding field/temperature handling, zero-field subtraction, and
saturation-magnetization analysis. VTVH_MCD further specializes it for
interval-scan-mode data with replicate averaging.
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import least_squares

from .abscd import AbsCD
from .labdata import DEFAULT_COLORS

# Default matplotlib color cycle for the satmag plots. A module-level tuple rather
# than a mutable list default, which the plotting code used to extend in place.
PLT_COLORS = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
)

# Some global variable
NM = "NANOMETERS"
WAVE = "Wavenums"
ABS = "ABSORBANCE"
CD = "CD/DC [mdeg]"
CD_INTERVAL = "CD [mdeg]"
TEMP = "SampleTemp_SetPt(K)"
FIELD = "Field_SetPoint(T)"
TEMP_AVG = "Avg_Sample_Temp(K)"
FIELD_AVG = "Avg_Magnet_Field(T)"
TEMP_DEV = "StdDev_Sample_Temp(K)"
FIELD_DEV = "StdDev_Magnet_Field(T)"


class MCD(AbsCD):
    """MCD Data Class for data from the J-1700 with parent AbsCD"""

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
        """See LabData.__init__ for the shared arguments. labels defaults to the
        standard J-1700 MCD column names if not supplied."""
        # Add dict for df column label abbreviations
        if labels is None:
            labels = {
                "NM": "NANOMETERS",
                "WAVE": "Wavenums",
                "ABS": "ABSORBANCE",
                "CD": "CD/DC [mdeg]",
                "CD_DEV": None,
                "TEMP": "SampleTemp_SetPt(K)",
                "FIELD": "Field_SetPoint(T)",
                "TEMP_AVG": "Avg_Sample_Temp(K)",
                "FIELD_AVG": "Avg_Magnet_Field(T)",
                "TEMP_DEV": "StdDev_Sample_Temp(K)",
                "FIELD_DEV": "StdDev_Magnet_Field(T)",
                "SCAN_NUM": "Scan_Num",
            }

        super().__init__(
            experiment_df,
            info_csv,
            info_df,
            processing_metadata,
            path_to_raw_data,
            labels,
            **kwargs,
        )
        # AbsCD.__init__ already drops the unnamed columns

    def process(self, path_to_raw_data, info_csv, **kwargs):
        super().process(path_to_raw_data, info_csv, **kwargs)
        if "id" not in self.info_df.columns:
            ids = []
            for i, row in self.info_df.iterrows():
                scan = row[self.labels["SCAN_NUM"]]
                temp = row[self.labels["TEMP"]]
                field = row[self.labels["FIELD"]]
                ids.append(f"{scan}_{temp:.1f}K_{field:.1f}T")
            self.info_df["id"] = ids
        return True

    def update_labels(self, new_labels):
        """Update the label abbreviations for columns with a new dict."""
        for key in new_labels:
            self.labels[key] = new_labels[key]
        return self.labels

    def subtract(
        self,
        ref_id=-1,
        ys=None,
        ids_to_subtract=None,
        drop_zeros=True,
        same_temp=True,
        sub_once=True,
        ystd_col=None,
    ):
        """Subtract off 0 T MCD spectra from the correct nonzero field spectra.
        ref_id - dictates which zero you will be using, -1 by default is the 0T before, +1 is the 0T after, else takes id of the zero you want.
        ids_to_subtract - which spectra to subtract off of, by default is all of them
        ys - name of column(s) in data to subtract, defaults to self.labels["CD"]"""
        # Look for y column
        if ys is None:
            ys = self.labels["CD"]

        # Get zeros
        zero_idx = self.info_df.index[self.info_df[self.labels["FIELD"]] == 0].to_list()
        # Create a new column to log subtractions
        if "Subtracted" not in self.info_df.columns:
            self.info_df["Subtracted"] = None

        # See which spectra to sub with each zero with builtin logic
        if ref_id in [-1, 1]:
            for zi in range(len(zero_idx)):
                # sub_id resets each iteration: when it lived outside the loop, a zero
                # that selected nothing silently re-subtracted the previous zero's list
                sub_id = None
                # Set a condition for scans having same temp to subtract them
                if same_temp:
                    temp_bool = (
                        self.info_df[self.labels["TEMP"]]
                        == self.info_df.at[zero_idx[zi], self.labels["TEMP"]]
                    )
                    # print(self.info_df.loc[temp_bool])
                    # Find the next/previous matching 0T to use as boundary condition for search
                    next_zero = False
                    for zj in range(zi + 1, len(zero_idx)):
                        if (
                            self.info_df.at[zero_idx[zj], self.labels["TEMP"]]
                            == self.info_df.at[zero_idx[zi], self.labels["TEMP"]]
                        ):
                            next_zero = self.info_df.at[
                                zero_idx[zj], self.labels["SCAN_NUM"]
                            ]
                            break
                        else:
                            pass  # keep looking

                    prev_zero = False
                    for zj in range(zi - 1, -1, -1):
                        if (
                            self.info_df.at[zero_idx[zj], self.labels["TEMP"]]
                            == self.info_df.at[zero_idx[zi], self.labels["TEMP"]]
                        ):
                            prev_zero = self.info_df.at[
                                zero_idx[zj], self.labels["SCAN_NUM"]
                            ]
                            break
                        else:
                            pass  # keep looking
                else:
                    temp_bool = self.info_df[self.labels["SCAN_NUM"]].notnull()
                    # guard the ends of the list: zi+1 ran off the end on the last zero,
                    # and zi-1 wrapped around to the last zero on the first one
                    next_zero = (
                        self.info_df.at[zero_idx[zi + 1], self.labels["SCAN_NUM"]]
                        if zi + 1 < len(zero_idx)
                        else False
                    )
                    prev_zero = (
                        self.info_df.at[zero_idx[zi - 1], self.labels["SCAN_NUM"]]
                        if zi > 0
                        else False
                    )

                if ref_id == -1:
                    # select ones after the zero (and before the next 0T)
                    if zi < len(zero_idx) - 1 and next_zero is not False:
                        sub_id = self.info_df.loc[
                            (self.info_df[self.labels["FIELD"]] != 0)
                            & (
                                self.info_df[self.labels["SCAN_NUM"]]
                                > self.info_df.at[zero_idx[zi], self.labels["SCAN_NUM"]]
                            )
                            & (self.info_df[self.labels["SCAN_NUM"]] < next_zero)
                            & (temp_bool)
                        ]["id"].to_numpy()
                    else:
                        sub_id = self.info_df.loc[
                            (self.info_df[self.labels["FIELD"]] != 0)
                            & (
                                self.info_df[self.labels["SCAN_NUM"]]
                                > self.info_df.at[zero_idx[zi], self.labels["SCAN_NUM"]]
                            )
                            & (temp_bool)
                        ]["id"].to_numpy()

                elif ref_id == 1:
                    # select ones before the zero (and after the previous 0T)
                    if zi > 0 and prev_zero is not False:
                        sub_id = self.info_df.loc[
                            (self.info_df[self.labels["FIELD"]] != 0)
                            & (
                                self.info_df[self.labels["SCAN_NUM"]]
                                < self.info_df.at[zero_idx[zi], self.labels["SCAN_NUM"]]
                            )
                            & (self.info_df[self.labels["SCAN_NUM"]] > prev_zero)
                            & (temp_bool)
                        ]["id"].to_numpy()
                    elif zi == 0:
                        sub_id = self.info_df.loc[
                            (self.info_df[self.labels["FIELD"]] != 0)
                            & (
                                self.info_df[self.labels["SCAN_NUM"]]
                                < self.info_df.at[zero_idx[zi], self.labels["SCAN_NUM"]]
                            )
                            & (temp_bool)
                        ]["id"].to_numpy()

                # nothing matched this zero (e.g. no scans at its temperature) -- skip it
                if sub_id is None or len(sub_id) == 0:
                    continue
                # Enforce that it can only get subtracted once
                if sub_once:
                    sub_id = self.info_df.loc[
                        (self.info_df["id"].isin(sub_id))
                        & (self.info_df["Subtracted"].isnull())
                    ]["id"].to_numpy()
                # Get id of the 0T for printing and then do the subtraction
                zid = self.info_df.at[zero_idx[zi], "id"]
                print(f"Subtracted {zid} off of {sub_id} for {ys}")
                super().subtract(ref_id=zid, ids_to_subtract=sub_id, ys=ys)
                # Handle error propogation
                if ystd_col is None:
                    ystd_col = self.labels["CD_DEV"]
                if ystd_col is not None:
                    super().add(ref_id=zid, ids_to_subtract=sub_id, ys=ystd_col)
                # Add label to subtracted column
                for sub_idx in self.get_idx_from_id(sub_id):
                    self.info_df.at[sub_idx, "Subtracted"] = zid

            # drop the zeros from the df
            if drop_zeros:
                print("Dropped Zeros from info_df.")
                self.info_df.drop(zero_idx, axis=0, inplace=True)

        # if only want to subtract one zero off of selected
        else:
            print(f"Subtracted {ref_id} off of {ids_to_subtract} for {ys}.")
            super().subtract(ref_id=ref_id, ids_to_subtract=ids_to_subtract, ys=ys)

            # Handle error propogation
            if ystd_col is None and ys == self.labels["CD"]:
                ystd_col = self.labels["CD_DEV"]
            if ystd_col is not None:
                print(f"Used {ystd_col} for Standard Deviation.")
                super().add(
                    ref_id=ref_id,
                    ids_to_subtract=ids_to_subtract,
                    ys=ystd_col,
                    square=True,
                )

            if drop_zeros:
                print(f"Dropped {ref_id} from info_df.")
                self.info_df.drop(
                    self.info_df.index[self.info_df["id"] == ref_id],
                    axis=0,
                    inplace=True,
                )
            for sub_idx in self.get_idx_from_id(ids_to_subtract):
                self.info_df.at[sub_idx, "Subtracted"] = ref_id

        # reset index so doesnt mess up other functions
        self.info_df.reset_index(inplace=True, drop=True)
        return True

    def subtract_component(
        self,
        component: AbsCD,
        ys=None,
        ystd_col=None,
        scalar: float = 1.0,
        inplace: bool = True,
    ):
        """Subtracts the data of component (multipiled by scalar) from scans of self taken at the same temp and field."""
        # Look for y column
        if ys is None:
            ys = self.labels["CD"]
            # cols_to_avg = [col for col in list(self.info_df.at[0,'data'].columns) if '_avg' in col]
            # cols_to_std = [col for col in list(self.info_df.at[0,'data'].columns) if '_std' in col]
        # Find Standard Deviation Column
        if ystd_col is None:
            ystd_col = self.labels["CD_DEV"]

        component_df = component.info_df.copy()
        # only pay for the deep copy when we actually need a separate object
        new_data = self if inplace else self.copy()
        # iterate through every combination of temp and field
        for i, row in self.info_df.iterrows():
            # select all rows with that t and f combo
            t = row[self.labels["TEMP"]]
            f = row[self.labels["FIELD"]]
            match_component = (
                component_df.loc[
                    (component_df[component.labels["TEMP"]] == t)
                    & (component_df[component.labels["FIELD"]] == f)
                ]
                .sort_values(component.labels["SCAN_NUM"])
                .copy()
            )
            if len(match_component) > 0:
                # select data
                match_component.reset_index(inplace=True)
                match_row = match_component.at[0, "data"]
                # print(f'{t}K_{f}T')

                # Update the data
                new_datadf = row["data"].copy()
                new_datadf[ys] = row["data"][ys].sub(match_row[ys].mul(scalar))

                if ystd_col is not None:
                    # CHANGED (Claude): was .mul(scalar), now .mul(scalar**2). The component
                    # spectrum is scaled by `scalar`, so its standard deviation scales by
                    # `scalar` and its variance -- which is what is being added here -- by
                    # scalar**2. The old form under-weighted the component's error for
                    # scalar > 1 and over-weighted it for scalar < 1; identical at scalar = 1.
                    new_datadf[ystd_col] = (
                        row["data"][ystd_col]
                        .pow(2)
                        .add(match_row[ystd_col].pow(2).mul(scalar**2))
                        .pow(0.5)
                    )

                new_data.info_df.at[i, "data"] = new_datadf.copy()

        return new_data

    def loc_field(self, field, inplace: bool = True, reset_index: bool = True):
        """Slice the info_df of the MCD Data object to only include specific fields.
        field = '+' for all positive fields and '-' for negative fields or an int/float or list of fields in T.
        Returns either a bool and edits dataframe inplace or returns new MCD object based on inplace parameter.
        """
        if field == "+":
            criteria = self.info_df[self.labels["FIELD"]] > 0
        elif field == "-":
            criteria = self.info_df[self.labels["FIELD"]] < 0
        elif isinstance(field, (int, float)):
            criteria = self.info_df[self.labels["FIELD"]] == field
        elif isinstance(field, list):
            criteria = self.info_df[self.labels["FIELD"]].isin(field)
        else:
            print("Did not pass field parameter correctly.")
            return False
        # slice the dataframe in place or into a new mcd object
        if inplace:
            # .copy() so later .at[] writes hit a real frame, not a slice view
            self.info_df = self.info_df.loc[criteria].copy()
            if reset_index:
                self.info_df.reset_index(inplace=True, drop=True)
            return True
        else:
            newmcd = self.copy()
            newmcd.info_df = self.info_df.loc[criteria].copy()
            if reset_index:
                newmcd.info_df.reset_index(inplace=True, drop=True)
            return newmcd

    def check_mirroring(self, field=None, plot=True, plot_x=NM, **kwargs):
        """Do +T + -T (should be subtracted from 0 already) which should go to 0 to check for mirroring.
        Takes field as an input of either float/int or list if only want to check specific fields in the dataset.
        """

        sub_data = self.copy()

        # Select the correct spectra to compare
        if field is None:
            sub_data.info_df = sub_data.info_df.loc[
                sub_data.info_df[self.labels["FIELD"]] > 0
            ].copy()
        elif isinstance(field, (float, int)):
            sub_data.info_df = sub_data.info_df.loc[
                sub_data.info_df[self.labels["FIELD"]] == field
            ].copy()
        elif isinstance(field, list):
            sub_data.info_df = sub_data.info_df.loc[
                sub_data.info_df[self.labels["FIELD"]].isin(field)
            ].copy()

        # Find negative field of same magnitude
        oppo_idxs = {}
        for i, row in sub_data.info_df.iterrows():
            # Find the scan for opposite sign field of same magnitude
            oppo = (
                self.info_df.loc[
                    (
                        self.info_df[self.labels["FIELD"]]
                        == (-1) * row[self.labels["FIELD"]]
                    )
                    & (self.info_df[self.labels["TEMP"]] == row[self.labels["TEMP"]])
                ]
                .copy()
                .reset_index()
            )
            # if there are multiple "opposite fields" that match the criteria, find the one taken closest to its counterpart
            closest_idx = 0
            rowfield = row[self.labels["FIELD"]]
            rowtemp = row[self.labels["TEMP"]]
            oppo_idx_key = f"{rowfield}_{rowtemp}"
            if oppo_idx_key not in list(oppo_idxs.keys()):
                oppo_idxs[oppo_idx_key] = []
            while closest_idx in oppo_idxs[oppo_idx_key]:
                closest_idx = closest_idx + 1
            for j, row2 in oppo.iterrows():
                if abs(
                    row2[self.labels["SCAN_NUM"]] - row[self.labels["SCAN_NUM"]]
                ) <= abs(
                    oppo.at[closest_idx, self.labels["SCAN_NUM"]]
                    - row[self.labels["SCAN_NUM"]]
                ):
                    if (
                        len(oppo)
                        != len(
                            sub_data.info_df.loc[
                                (
                                    self.info_df[self.labels["FIELD"]]
                                    == row[self.labels["FIELD"]]
                                )
                                & (
                                    self.info_df[self.labels["TEMP"]]
                                    == row[self.labels["TEMP"]]
                                )
                            ]
                        )
                        or j not in oppo_idxs[oppo_idx_key]
                    ):  # only allow duplicates if lists not the same size
                        closest_idx = j
            # Track what has already been used
            if closest_idx not in oppo_idxs[oppo_idx_key]:
                oppo_idxs[oppo_idx_key].append(closest_idx)
            # select opposite field with correct index
            oppo = oppo.iloc[closest_idx]
            # print(closest_idx, row['id'], oppo_idxs)
            # Do the arithmetic
            sub_data.info_df.at[i, "data"][self.labels["CD"]] = np.add(
                row["data"][self.labels["CD"]], oppo["data"][self.labels["CD"]]
            )
            # Add the corresponding label
            sub_data.info_df.at[i, "id"] = row["id"] + " + " + oppo["id"]

        # Plot if desired
        if plot:
            fig = sub_data.quick_plot(y=self.labels["CD"], x=plot_x, **kwargs)
            return fig
        else:
            return sub_data

    def half_subd_fields(self, field=None, ystd_col=None, inplace: bool = True):
        """Do 0.5*(+T - -T) as a replacement for subtracting off zeroes
        field - takes a positive field or list of fields to do this for (default is all of them)
        (Currently might be an error if have duplicates of the same field & temp in finding the negative one)
        """
        sub_data = self.copy()

        # Select the correct spectra to compare
        if field is None:
            sub_data.info_df = sub_data.info_df.loc[
                sub_data.info_df[self.labels["FIELD"]] > 0
            ].copy()
        elif isinstance(field, (float, int)):
            sub_data.info_df = sub_data.info_df.loc[
                sub_data.info_df[self.labels["FIELD"]] == field
            ].copy()
        elif isinstance(field, list):
            sub_data.info_df = sub_data.info_df.loc[
                sub_data.info_df[self.labels["FIELD"]].isin(field)
            ].copy()

        # Find negative field of same magnitude
        oppo_idxs = {}
        for i, row in sub_data.info_df.iterrows():
            # Find the scan for opposite sign field of same magnitude
            oppo = (
                self.info_df.loc[
                    (
                        self.info_df[self.labels["FIELD"]]
                        == (-1) * row[self.labels["FIELD"]]
                    )
                    & (self.info_df[self.labels["TEMP"]] == row[self.labels["TEMP"]])
                ]
                .copy()
                .reset_index()
            )
            # if there are multiple "opposite fields" that match the criteria, find the one taken closest to its counterpart
            closest_idx = 0
            rowfield = row[self.labels["FIELD"]]
            rowtemp = row[self.labels["TEMP"]]
            oppo_idx_key = f"{rowfield}_{rowtemp}"
            if oppo_idx_key not in oppo_idxs.keys():
                oppo_idxs[oppo_idx_key] = []
            while closest_idx in oppo_idxs[oppo_idx_key]:
                closest_idx = closest_idx + 1
            for j, row2 in oppo.iterrows():
                if abs(
                    row2[self.labels["SCAN_NUM"]] - row[self.labels["SCAN_NUM"]]
                ) <= abs(
                    oppo.at[closest_idx, self.labels["SCAN_NUM"]]
                    - row[self.labels["SCAN_NUM"]]
                ):
                    if (
                        len(oppo) != len(sub_data.info_df)
                        or j not in oppo_idxs[oppo_idx_key]
                    ):  # only allow duplicates if lists not the same size
                        closest_idx = j
            # Track what has already been used
            if closest_idx not in oppo_idxs[oppo_idx_key]:
                oppo_idxs[oppo_idx_key].append(closest_idx)
            # select opposite field with correct index
            if len(oppo) > 0:
                oppo = oppo.iloc[closest_idx]
                # Do the arithmetic to get avgs and std deviations
                sub_data.info_df.at[i, "data"][self.labels["CD"]] = 0.5 * np.subtract(
                    row["data"][self.labels["CD"]], oppo["data"][self.labels["CD"]]
                )
                if ystd_col is not None:
                    sub_data.info_df.at[i, "data"][ystd_col] = 0.5 * np.sqrt(
                        np.add(
                            np.square(row["data"][ystd_col]),
                            np.square(oppo["data"][ystd_col]),
                        )
                    )
                    # Mean of the variances + Variance of the means (https://arxiv.org/pdf/1007.1012) - DONT THINK THIS APPLIES HERE
                    # meanofvars = row['data'][ystd_col].pow(2).mul(row['n_replicates']).add(oppo['data'][ystd_col].pow(2).mul(oppo['n_replicates'])).div(row['n_replicates']+oppo['n_replicates'])
                    # varofmeans = row['data'][ystd_col.replace('_std','_avg')].mul(row['n_replicates']).add(oppo['data'][ystd_col.replace('_std','_avg')].mul(oppo['n_replicates'])).div(row['n_replicates']+oppo['n_replicates'])
                    # sub_data.info_df.at[i,'data'][ystd_col] = np.sqrt(np.add(meanofvars,varofmeans))

                sub_data.info_df.at[i, self.labels["TEMP_AVG"]] = np.mean(
                    [row[self.labels["TEMP_AVG"]], oppo[self.labels["TEMP_AVG"]]]
                )
                sub_data.info_df.at[i, self.labels["FIELD_AVG"]] = np.mean(
                    np.abs(
                        [row[self.labels["FIELD_AVG"]], oppo[self.labels["FIELD_AVG"]]]
                    )
                )
                sub_data.info_df.at[i, self.labels["TEMP_DEV"]] = 0.5 * np.sqrt(
                    np.add(
                        np.square(row[self.labels["TEMP_DEV"]]),
                        np.square(oppo[self.labels["TEMP_DEV"]]),
                    )
                )
                sub_data.info_df.at[i, self.labels["FIELD_DEV"]] = 0.5 * np.sqrt(
                    np.add(
                        np.square(row[self.labels["FIELD_DEV"]]),
                        np.square(oppo[self.labels["FIELD_DEV"]]),
                    )
                )
                # 'TEMP_AVG' : 'Avg_Sample_Temp(K)',
                # 'FIELD_AVG' : 'Avg_Magnet_Field(T)',
                # 'TEMP_DEV' : 'StdDev_Sample_Temp(K)',
                # 'FIELD_DEV' : 'StdDev_Magnet_Field(T)',
                # Add the corresponding label
                sub_data.info_df.at[i, "id"] = (
                    "0.5*(" + row["id"] + " - " + oppo["id"] + ")"
                )

        sub_data.info_df.reset_index(inplace=True, drop=True)

        if inplace:
            self.info_df = sub_data.info_df.copy()
        # Return the new MCD_Data object
        return sub_data

    def add_deps(self, conc, conc_units="M", path_length=0.3):
        """Add a y value of Delta Epsilon (1/M*cm) by converting from mdeg.
        conc - takes numerical value or name of dataframe column in self.data
        Defaults to pathlength = 0.3 cm for MCD"""
        super().add_deps(conc, conc_units=conc_units, path_length=path_length)

    def quick_plot(
        self,
        x=None,
        y=None,
        x_range=None,
        y_range=None,
        ids=None,
        height=None,
        width=1000,
        fig=None,
        colors=DEFAULT_COLORS,
        selection_criteria=None,
        field=None,
        temp=None,
        **kwargs,
    ):
        """LabData.quick_plot plus `field` and `temp` shortcuts, which build the
        selection_criteria mask for you from the field/temp set points (in T and K).
        A field/temp given here overrides any selection_criteria passed in."""
        if isinstance(field, (int, float)):
            field = [field]
        if isinstance(temp, (int, float)):
            temp = [temp]

        if field is not None and temp is None:
            selection_criteria = self.info_df[self.labels["FIELD"]].isin(field)
        elif temp is not None and field is None:
            selection_criteria = self.info_df[self.labels["TEMP"]].isin(temp)
        elif temp is not None and field is not None:
            selection_criteria = (self.info_df[self.labels["TEMP"]].isin(temp)) & (
                self.info_df[self.labels["FIELD"]].isin(field)
            )
        # keyword args so this does not silently break if the parent's parameter order changes
        return super().quick_plot(
            x=x,
            y=y,
            x_range=x_range,
            y_range=y_range,
            ids=ids,
            height=height,
            width=width,
            fig=fig,
            colors=colors,
            selection_criteria=selection_criteria,
            **kwargs,
        )

    def add_satmag_x(self):
        """Inserts a column into info_df for uBHover2kbT used as the x-axis in saturation magnetization curves.
        Uses the average field and temp columns."""
        # some initial useful constants from Wes
        zeemanFactor = 0.4668644735835207  # Bohr magneton in units of cm-1/Tesla
        RecipBoltz = 1.4387768775039338  # 1/kB in units of Kelvin/cm-1
        self.info_df["uBHover2kbT"] = (
            0.5
            * RecipBoltz
            * np.divide(
                np.multiply(zeemanFactor, abs(self.info_df[self.labels["FIELD_AVG"]])),
                self.info_df[self.labels["TEMP_AVG"]],
            )
        )
        self.info_df["1overkbT"] = RecipBoltz * np.divide(
            1, self.info_df[self.labels["TEMP_AVG"]]
        )

    def add_satmag_y(self, x_val: float, x_col: str = WAVE, y_col=None, ystd_col=None):
        """Inserts a column into info_df of normalized intensity of spectra at x_val (units of x_col)."""
        # assign default y_col value
        if y_col is None:
            y_col = self.labels["CD"]
        # hoisted out of the loop -- this only needs resolving once
        if ystd_col is None:
            ystd_col = self.labels["CD_DEV"]
        inten_x_val = []
        inten_x_std = np.zeros(len(self.info_df))
        # iterate through to save intensity for each scan
        true_xval = set()
        # enumerate for the positional write into inten_x_std: `i` is an info_df label,
        # which stops matching the array offset once rows are dropped without a reindex
        for pos, (i, row) in enumerate(self.info_df.iterrows()):
            # first point past x_val, computed once instead of three times
            nearest = (
                row["data"]
                .loc[row["data"][x_col] > x_val]
                .sort_values(x_col, ascending=True)
                .reset_index()
            )
            if len(nearest) == 0:
                # otherwise this fails with a bare KeyError: 0 several frames down
                raise ValueError(
                    f'{row["id"]} has no {x_col} value above x_val={x_val}; '
                    f"is x_val in the units of {x_col}?"
                )
            true_xval.add(nearest.at[0, x_col])
            inten_x_val.append(nearest.at[0, y_col])
            # add standard deviation at that point
            if ystd_col is not None:
                inten_x_std[pos] = nearest.at[0, ystd_col]
        # save the intensities and normalize to absolute value of the max
        # should probably error propogate based on this scaling
        print(f"Actual x_value: {true_xval} {x_col}")
        self.info_df[f"normI_{x_val:.1f}"] = inten_x_val
        self.info_df[f"I_{x_val:.1f}"] = inten_x_val
        norm_factor = max(np.abs(inten_x_val))
        self.info_df[f"normI_{x_val:.1f}"] = np.divide(
            abs(self.info_df[f"normI_{x_val:.1f}"]), norm_factor
        )
        self.info_df[f"stdI_{x_val:.1f}"] = np.divide(inten_x_std, norm_factor)

    def make_satmag_fit_csv(
        self,
        x_val,
        x_col: str = WAVE,
        y_col=None,
        name: str = "mcdfit_data.csv",
        ystd_col=None,
        save: bool = True,
    ) -> pd.DataFrame:
        """Generate the csv file needed to plug into Wes's VTVH Particle Swarm Fitting Code."""
        # assign default y_col value
        if y_col is None:
            y_col = self.labels["CD"]
        # Check for columns needed to make the plot
        if "uBHover2kbT" not in self.info_df.columns:
            self.add_satmag_x()
        fit_df = pd.DataFrame()
        fit_df["Temp"] = self.info_df[self.labels["TEMP_AVG"]]
        fit_df["Field"] = abs(self.info_df[self.labels["FIELD_AVG"]])

        if isinstance(x_val, (int, float)):
            x_val = [x_val]
        for i, x_vali in enumerate(x_val):
            if f"normI_{x_vali:.1f}" not in self.info_df.columns:
                self.add_satmag_y(
                    x_val=x_vali, x_col=x_col, y_col=y_col, ystd_col=ystd_col
                )

            fit_df[f"Data {i+1}"] = self.info_df[f"normI_{x_vali:.1f}"]
            fit_df[f"Error {i+1}"] = self.info_df[f"stdI_{x_vali:.1f}"]
        # write to csv
        if save:
            fit_df.to_csv(name, index=False, header=True)
        # return the df
        return fit_df

    def make_simpoints_csv(
        self, file_name: str = "vtvh_simpoints.csv", nfields: int = 100
    ):
        """Generate the csv file needed to plug into Wes's VTVH Simulation Code."""
        # generate list of the average temp for each set point
        avg_temps = []
        for i, temp in enumerate(self.info_df[self.labels["TEMP"]].unique()):
            avg_temps.append(
                self.info_df.loc[self.info_df[self.labels["TEMP"]] == temp][
                    self.labels["TEMP_AVG"]
                ].mean()
            )
        # generate list of fields
        fields = np.linspace(0, 7, nfields)
        # create lists of combinations
        hs, ts = np.meshgrid(fields, avg_temps)
        ts = np.concatenate(ts)
        hs = np.concatenate(hs)
        # send points to dataframe and output to csv
        simpoints = pd.DataFrame({"Temp": ts, "Field": hs})
        simpoints.to_csv(file_name, index=False, header=True)
        return simpoints

    def plt_satmag(
        self,
        x_val: float,
        iso: str = "temp",
        which_iso: list | None = None,
        x_col_unit: str = "$cm^{-1}$",
        x_col: str = WAVE,
        y_col=None,
        ystd_col=None,
        line=False,
        fig=None,
        ax=None,
        marker=".",
        separate=False,
        sharex=True,
        ncols=1,
        colors=PLT_COLORS,
        fontdict=None,
        **kwargs,
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Generate a matplotlib scatter plot of the saturation magnetization behavior at x_val x_col_unit (cm^-1)
        with either isotherms on a uBHover2kbT plot or isofields against 1overkbT."""
        if fontdict is None:
            fontdict = {}
        # handle y col
        if y_col is None:
            y_col = self.labels["CD"]
        if ystd_col is None:
            ystd_col = self.labels["CD_DEV"]
        # "field or temp or column name"
        # Check for columns needed to make the plot
        if "uBHover2kbT" not in self.info_df.columns:
            self.add_satmag_x()
        if f"normI_{x_val:.1f}" not in self.info_df.columns:
            self.add_satmag_y(x_val=x_val, x_col=x_col, y_col=y_col, ystd_col=ystd_col)
        if iso == "field":
            iso_col = self.labels["FIELD"]
            iso_unit = "T"
            satmag_x_col = "1overkbT"
            noniso_col = self.labels["TEMP"]
        elif iso == "temp":
            iso_col = self.labels["TEMP"]
            iso_unit = "K"
            satmag_x_col = "uBHover2kbT"
            noniso_col = self.labels["FIELD"]
        else:
            print(f"Using {iso} as column for lines. Does not work yet.")
            iso_col = iso
            iso_unit = iso
            satmag_x_col = None
            noniso_col = None

        iso_lines = abs(self.info_df[iso_col]).unique()
        # cycle into a fresh local list. The old `colors.extend(colors)` mutated the
        # default argument, so the default palette grew permanently on every call
        # that needed more colors than it had.
        colors = [
            colors[j % len(colors)] for j in range(max(len(iso_lines), len(colors)))
        ]
        # set up plot
        if fig is None or ax is None:
            if separate:
                fig, ax = plt.subplots(
                    ncols=ncols,
                    nrows=math.ceil(len(iso_lines) / ncols),
                    figsize=[4 * ncols, 2.5 * len(iso_lines) / ncols],
                    sharex=sharex,
                    squeeze=False,
                )
                ax = ax.flatten()
            else:
                fig, ax = plt.subplots(figsize=[5, 4])

        for j, il in enumerate(iso_lines):
            if which_iso is None or il in which_iso:
                sele_df = (
                    self.info_df.loc[abs(self.info_df[iso_col]) == il]
                    .copy()
                    .sort_values(noniso_col)
                )
                if separate:
                    ax[j].errorbar(
                        sele_df[satmag_x_col],
                        sele_df[f"normI_{x_val:.1f}"],
                        yerr=sele_df[f"stdI_{x_val:.1f}"],
                        label=f"{il} {iso_unit}",
                        linestyle="-" if line else "none",
                        marker=marker,
                        capsize=2,
                        color=colors[j],
                        **kwargs,
                    )
                    ax[j].legend()
                else:
                    ax.errorbar(
                        sele_df[satmag_x_col],
                        sele_df[f"normI_{x_val:.1f}"],
                        yerr=sele_df[f"stdI_{x_val:.1f}"],
                        label=f"{il} {iso_unit}",
                        linestyle="-" if line else "none",
                        marker=marker,
                        capsize=2,
                        color=colors[j],
                        **kwargs,
                    )
                # ax.errorbar(self.info_df.loc[abs(self.info_df[iso_col]) == il][satmag_x_col],self.info_df.loc[abs(self.info_df[iso_col]) == il][f'normI_{x_val:.1f}'],
                #         yerr=self.info_df.loc[abs(self.info_df[iso_col]) == il][f'stdI_{x_val:.1f}'], label=f'{il} {iso_unit}',
                #         linestyle='-' if line else 'none', marker=marker, capsize=2, **kwargs)
            # ax.plot(self.info_df.loc[abs(self.info_df[iso_col]) == il][satmag_x_col],self.info_df.loc[abs(self.info_df[iso_col]) == il][f'normI_{x_val:.1f}'])

        if not separate:
            if satmag_x_col == "uBHover2kbT":
                ax.set_xlabel(r"$\mu_B H / 2 k_B T$", fontdict=fontdict)
            else:
                ax.set_xlabel(f"{satmag_x_col}", fontdict=fontdict)
            ax.set_ylabel("Normalized MCD Intensity", fontdict=fontdict)
            ax.set_title(rf"{x_val} {x_col_unit}", fontdict=fontdict)
            ax.legend(prop=fontdict)
        return fig, ax

    # Function generated by ChatGPT - needs to be checked!!
    def calculate_I_MCD(self, g_parallel, delta, A_SatLim, H, T) -> float:
        """A function to calculate the MCD intensity using the doublet model."""
        # Constants
        beta = 0.4668644735835207
        k = 1 / 1.4387768775039338  # Calculate k from the given constant

        # Calculate thermal energy
        kT = k * T

        def integrand(theta):
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            term1 = cos_theta**2 * sin_theta
            term2 = np.sqrt(delta**2 + (g_parallel * beta * H * cos_theta) ** 2)
            term3 = g_parallel * beta * H
            tanh_term = np.tanh(
                np.sqrt(delta**2 + (g_parallel * beta * H * cos_theta) ** 2) / (2 * kT)
            )

            return (term1 / term2) * term3 * tanh_term

        # Perform the integration from 0 to π/2
        integral_result, _ = quad(integrand, 0, np.pi / 2)

        # Calculate I_MCD
        I_MCD = A_SatLim * integral_result
        return I_MCD

    # This code was generated with the help of ChatGPT.

    def calculate_I_MCD_wPolarizations(
        self,
        fitvars: list[float],
        H: float,
        T: float,
        g_perpendicular: float = 1.0,
        g_parallel=8.0,
        delta=0,
        MzoverMxy=0,
        B0=0,
        exponential=False,
        neg_M=True,
    ) -> float:
        r"""
        A function to calculate the MCD intensity using the doublet model now including z polarization and B term.
        fitvars in form [A_satlim, g_parallel, delta, MzoverMxy, B0]. delta, MzoverMxy, and B0 optional
        Calculate \( I_{MCD} \) considering polarizations based on the provided parameters.

        The equation being solved is:
        \[
        Δε = A_{satlim} \left\{ \int_0^{\frac{\pi}{2}} \left( \frac{g_{\parallel} \beta H}{Γ} \right) \cos^2 θ \sin θ \tanh\left( \frac{Γ}{2kT} \right) dθ
        - \sqrt{2} \left( \frac{M_z}{M_{xy}} \right) \int_0^{\frac{\pi}{2}} \left( \frac{g_{\perp} \beta H}{Γ} \right) \sin^3 θ \tanh\left( \frac{Γ}{2kT} \right) dθ \right\}
        + B_0 H

        Where:
        \[
        Γ = \sqrt{δ^2 + (g_{\parallel} β H \cos θ)^2 + (g_{\perp} β H \sin θ)^2}
        \]
        """
        # Set variables
        if exponential:
            A_SatLim = np.exp(fitvars[0])
        else:
            A_SatLim = fitvars[0]

        if len(fitvars) > 1:
            if exponential:
                g_parallel = np.exp(fitvars[1])
            else:
                g_parallel = fitvars[1]
        if len(fitvars) > 2:
            if exponential:
                delta = np.exp(fitvars[2])
            else:
                delta = fitvars[2]
        if len(fitvars) > 3:
            if exponential:
                MzoverMxy = -1 * np.exp(fitvars[3]) if neg_M else np.exp(fitvars[3])
            else:
                MzoverMxy = fitvars[3]
        if len(fitvars) > 4:
            if exponential:
                B0 = np.exp(fitvars[4])
            else:
                B0 = fitvars[4]

        # Constants
        beta = 0.4668644735835207
        k = 1 / 1.4387768775039338  # Calculate k from the given constant

        # Calculate thermal energy
        kT = k * T

        def integrand_1(theta: float) -> float:
            r"""
            Integrand for the first integral:
            \left( \frac{g_{\parallel} \beta H}{Γ} \right) \cos^2 θ \sin θ \tanh\left( \frac{Γ}{2kT} \right)
            """
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            gamma = np.sqrt(
                delta**2
                + (g_parallel * beta * H * cos_theta) ** 2
                + (g_perpendicular * beta * H * sin_theta) ** 2
            )
            term = (
                (g_parallel * beta * H / gamma)
                * cos_theta**2
                * sin_theta
                * np.tanh(gamma / (2 * kT))
            )
            return term

        def integrand_2(theta: float) -> float:
            r"""
            Integrand for the second integral:
            \left( \frac{g_{\perp} \beta H}{Γ} \right) \sin^3 θ \tanh\left( \frac{Γ}{2kT} \right)
            """
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            gamma = np.sqrt(
                delta**2
                + (g_parallel * beta * H * cos_theta) ** 2
                + (g_perpendicular * beta * H * sin_theta) ** 2
            )
            term = (
                (g_perpendicular * beta * H / gamma)
                * sin_theta**3
                * np.tanh(gamma / (2 * kT))
            )
            return term

        # Integrate the first part
        integral_1, _ = quad(integrand_1, 0, np.pi / 2)

        # Integrate the second part
        integral_2, _ = quad(integrand_2, 0, np.pi / 2)

        # Calculate Δε
        delta_e = A_SatLim * (integral_1 - np.sqrt(2) * MzoverMxy * integral_2) + B0 * H
        return delta_e

    def resid_doublet_vtvh(self, fitvars, Hs, Ts, mcd_inten, **kwargs):
        """Residual Calculator for Fitting VTVH Data to I_MCD Doublet Model.
        fitvars in form [A_satlim, g_parallel, delta, MzoverMxy, B0].
        Returns one signed residual per data point, which is the form least_squares expects.
        """
        # Check input list length
        if not (len(Hs) == len(Ts) and len(Hs) == len(mcd_inten)):
            raise ValueError("Length of Input Lists not the same.")
        # to_numpy so positional indexing works even when these arrive as pandas Series
        # with a non-zero-based index
        Hs = np.asarray(Hs)
        Ts = np.asarray(Ts)
        mcd_inten = np.asarray(mcd_inten)
        # CHANGED (Claude): was `total_resid += abs(model - data)`, returning one summed
        # scalar. least_squares minimizes 0.5*sum(f_i**2) over the *vector* f it gets back
        # and estimates the Jacobian column-by-column from it, so a single scalar gave it
        # a 1-by-nparams Jacobian -- an underdetermined, badly conditioned problem where
        # most parameter directions looked flat. Returning the per-point vector is the
        # standard form. Signed rather than abs(): the sum of squares is identical either
        # way, but abs() puts a kink at zero that breaks the finite-difference Jacobian.
        # Expect different (better-converged) parameters than earlier fits.
        resids = np.empty(len(Hs), dtype=float)
        for i in range(len(Hs)):
            # resids[i] = self.calculate_I_MCD(fitvars[0], fitvars[1], fitvars[2], Hs[i], Ts[i]) - mcd_inten[i]
            resids[i] = (
                self.calculate_I_MCD_wPolarizations(fitvars, Hs[i], Ts[i], **kwargs)
                - mcd_inten[i]
            )
        return resids

    def fit_doublet_model_vtvh(
        self,
        fitvars: list[float],
        x_val: float,
        x_col: str = WAVE,
        bounds=(-1 * np.inf, np.inf),
        gtol=1e-13,
        ftol=1e-13,
        xtol=1e-13,
        max_nfev=None,
        y_col=None,
        ystd_col=None,
        verbose=1,
        **kwargs,
    ):
        """Fitting function to fit VTVH Data to g_parallel and delta of the doublet model.
        fitvars in form [A_satlim, g_parallel, delta, MzoverMxy, B0]"""
        # assign default y_col value
        if y_col is None:
            y_col = self.labels["CD"]
        if f"normI_{x_val:.1f}" not in self.info_df.columns:
            self.add_satmag_y(x_val=x_val, x_col=x_col, y_col=y_col, ystd_col=ystd_col)

        # run fit
        fit = least_squares(
            self.resid_doublet_vtvh,
            fitvars,
            bounds=bounds,
            args=(
                abs(self.info_df[self.labels["FIELD_AVG"]]).to_numpy(),
                self.info_df[self.labels["TEMP_AVG"]].to_numpy(),
                self.info_df[f"normI_{x_val:.1f}"].to_numpy(),
            ),
            kwargs=kwargs,
            verbose=verbose,
            gtol=gtol,
            ftol=ftol,
            xtol=xtol,
            max_nfev=max_nfev,
        )
        return fit

    def plot_doublet_model_vtvh(
        self,
        fitvars,
        x_val: float,
        x_col: str = WAVE,
        colors=PLT_COLORS,
        separate=False,
        **kwargs,
    ):
        """Plotting function to overlay the fitted doublet model on the VTVH data.
        fitvars in form [A_satlim, g_parallel, delta, MzoverMxy, B0]"""
        if f"normI_{x_val:.1f}" not in self.info_df.columns:
            self.add_satmag_y(x_val=x_val, x_col=x_col)

        fig, ax = self.plt_satmag(
            x_val=x_val,
            iso="temp",
            x_col=x_col,
            separate=separate,
            colors=colors,
            **kwargs,
        )

        # Code designates for temp iso plot from above
        iso_col = self.labels["TEMP"]
        iso_unit = "K"
        # satmag_x_col = 'uBHover2kbT'
        # noniso_col= self.labels['FIELD']
        iso_lines = abs(self.info_df[iso_col]).unique()
        fields = np.linspace(0, 7, 100)
        # some initial useful constants from Wes
        zeemanFactor = 0.4668644735835207  # Bohr magneton in units of cm-1/Tesla
        RecipBoltz = 1.4387768775039338  # 1/kB in units of Kelvin/cm-1

        for i, temp in enumerate(iso_lines):
            avg_temp = self.info_df.loc[self.info_df[self.labels["TEMP"]] == temp][
                self.labels["TEMP_AVG"]
            ].mean()
            plot_xvals = (
                0.5
                * RecipBoltz
                * np.divide(np.multiply(zeemanFactor, fields), avg_temp)
            )
            # fit_inten = [self.calculate_I_MCD(g_parallel=fitvars[0], delta=fitvars[1], A_SatLim=fitvars[2], H=h, T=temp) for h in fields]
            fit_inten = [
                self.calculate_I_MCD_wPolarizations(fitvars, H=h, T=avg_temp)
                for h in fields
            ]

            if separate:
                ax[i].plot(
                    plot_xvals, fit_inten, label=f"{avg_temp} {iso_unit}", c=colors[i]
                )
            else:
                ax.plot(
                    plot_xvals, fit_inten, label=f"{avg_temp} {iso_unit}", c=colors[i]
                )
            # ax.legend()

        return fig, ax


class VTVH_MCD(MCD):
    """VTVH MCD Data Class for data from the J-1700 in interval scan mode with parent MCD.

    Interval-scan files hold several replicate channels per scan, so the default
    labels point at the averaged/std columns that load() builds.
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
        # Add dict for df column label abbreviations
        if labels is None:
            labels = {
                "NM": "NANOMETERS",
                "WAVE": "Wavenums",
                "ABS": "ABSORBANCE",
                "CD": "CD/DC [mdeg]_avg",
                "CD_DEV": "CD/DC [mdeg]_std",
                "TEMP": "SampleTemp_SetPt(K)",
                "FIELD": "Field_SetPoint(T)",
                "TEMP_AVG": "Avg_Sample_Temp(K)",
                "FIELD_AVG": "Avg_Magnet_Field(T)",
                "TEMP_DEV": "StdDev_Sample_Temp(K)",
                "FIELD_DEV": "StdDev_Magnet_Field(T)",
                "SCAN_NUM": "Scan_Num",
            }

        super().__init__(
            experiment_df,
            info_csv,
            info_df,
            processing_metadata,
            path_to_raw_data,
            labels,
            **kwargs,
        )

    def load(self, path_to_raw_data, data_files, j1700=True, verbose=False, **kwargs):
        """Convert from raw csv files to standardized (specifically for J-1700 Abs/CD in Interval Scan Mode).
        If data was saved using the code and not directly by the J-1700 set j1700=False to use parent function.
        """
        if not j1700:
            # Use parent load function if saved data from code rather than j1700
            print("Using Parent Class Load function.")
            return super().load(path_to_raw_data, data_files, j1700=False, **kwargs)

        # use pandas to read in and aggregate data files
        # parse data from each J-1700 data file
        xlabels = []
        ylabels = []
        # go through the data files
        for i, row in self.info_df.iterrows():
            if self.info_df.at[i, "File"] in data_files:
                if verbose:
                    print(self.info_df.at[i, "File"])
                file_path = os.path.join(path_to_raw_data, self.info_df.at[i, "File"])
                # open and read each file to extract necessary infomation for reading with pandas
                # npts resets per file so a blank NPOINTS line cannot reuse the previous file's count
                npts = None
                with open(file_path, "r") as f:
                    count = 1
                    for line in f:
                        if "UNITS" in line and "X" in line:
                            xstr = line.split(",")[-1].replace("\n", "")
                            if xstr not in xlabels:
                                xlabels.append(xstr)
                        elif "UNITS" in line and "Y" in line:
                            ystr = line.split(",")[-1].replace("\n", "")
                            if ystr not in ylabels:
                                ylabels.append(ystr)
                        elif "NPOINTS" in line:
                            if line.split(",")[-1] != "\n":
                                npts = int(line.split(",")[-1])
                            else:
                                print(line)
                        elif "Channel" in line:
                            break
                        count = count + 1
                if npts is None:
                    raise ValueError(
                        f"No NPOINTS value found in {file_path}; is this an interval-scan export?"
                    )
                # read in csv as dataframe to correct place in info_df
                self.info_df.at[i, "data"] = pd.DataFrame()
                for yi, channel in enumerate(ylabels):
                    # read in df from single channel
                    channel_df = pd.read_csv(
                        file_path,
                        skiprows=count,
                        header=0,
                        index_col=False,
                        na_values="1.#INF",
                        nrows=npts,
                        dtype=float,
                    )
                    # rename columns appropriately
                    new_names = [
                        f"{channel}_{k}" if k != 0 else xlabels[0]
                        for k, name in enumerate(list(channel_df.columns))
                    ]
                    new_names_dict = {}
                    for old, new in zip(list(channel_df.columns), new_names):
                        new_names_dict[old] = new
                    channel_df.rename(columns=new_names_dict, inplace=True)

                    # Include x label in the first channel only. The old `channel_df.drop(...)`
                    # here had no inplace/assignment, so it was a no-op; assign the result.
                    if yi != 0:
                        channel_df = channel_df.drop(xlabels[0], axis=1)
                    # Add new channel df to overall data df
                    self.info_df.at[i, "data"][list(channel_df.columns)] = channel_df[
                        list(channel_df.columns)
                    ]

                    count += (
                        npts + 2
                    )  # increase rows to skip to include the next table of values + "channel" header and data header
        # Do avg and std of data
        for y_str in ylabels:
            self._average_ys(y_str=y_str)
        return True

    def _average_ys(self, y_str: str = CD_INTERVAL):
        """Adds an average and standard deviation column for the y values that contain the string specified as y_str."""
        # Find col names to use
        cols_to_avg = [
            col for col in list(self.info_df["data"].iloc[0].columns) if y_str in col
        ]
        # have col to keep track of total replicates
        self.info_df["n_replicates"] = len(cols_to_avg)
        # Make a new avg column
        # iterate the index labels rather than range(len(...)), which assumes a 0..n-1 index
        for i in self.info_df.index:
            # Just for troubleshooting if something is wrong
            if isinstance(self.info_df.at[i, "data"], float):
                print(self.info_df.at[i, "File"])
                print(self.info_df.at[i, "data"])

            # actual averaging operations
            self.info_df.at[i, "data"][y_str + "_avg"] = self.info_df.at[i, "data"][
                cols_to_avg
            ].mean(axis=1)
            self.info_df.at[i, "data"][y_str + "_std"] = self.info_df.at[i, "data"][
                cols_to_avg
            ].std(axis=1)
            self.info_df.at[i, "data"].drop(columns=cols_to_avg, inplace=True)

    def average_replicates(self):
        """Average the various parameters and data of scans taken at the same temp and field."""
        # Compile all the unique temps and fields
        all_temps = self.info_df[self.labels["TEMP"]].unique()
        all_fields = self.info_df[self.labels["FIELD"]].unique()
        new_info_df = pd.DataFrame()
        # new_info_df['data'] = pd.Series(dtype='object')
        new_data_dfs = []
        # iterate through every combination of temp and field
        for t in all_temps:
            for f in all_fields:
                # select all rows with that t and f combo
                tf_df = (
                    self.info_df.loc[
                        (self.info_df[self.labels["TEMP"]] == t)
                        & (self.info_df[self.labels["FIELD"]] == f)
                    ]
                    .sort_values(self.labels["SCAN_NUM"])
                    .copy()
                )
                if len(tf_df) > 0:
                    tf_df.reset_index(inplace=True)
                    # print(f'{t}K_{f}T')
                    # Update the values in the info_df.
                    # Built as a dict and converted once at the end: growing an empty
                    # pd.Series key-by-key reallocates each time and warns about dtype.
                    new_row = {}
                    new_row["Date"] = tf_df.at[0, "Date"]
                    new_row[self.labels["SCAN_NUM"]] = tf_df.at[
                        0, self.labels["SCAN_NUM"]
                    ]
                    new_row[self.labels["TEMP"]] = tf_df.at[0, self.labels["TEMP"]]
                    new_row[self.labels["FIELD"]] = tf_df.at[0, self.labels["FIELD"]]
                    new_row[self.labels["TEMP_AVG"]] = tf_df[
                        self.labels["TEMP_AVG"]
                    ].mean(axis=0)
                    new_row[self.labels["FIELD_AVG"]] = tf_df[
                        self.labels["FIELD_AVG"]
                    ].mean(axis=0)
                    # new_row[self.labels['TEMP_DEV']] = 0.5* np.sqrt(np.sum(np.square(tf_df[self.labels['TEMP_DEV']])))
                    # new_row[self.labels['FIELD_DEV']] = 0.5* np.sqrt(np.sum(np.square(tf_df[self.labels['FIELD_DEV']])))
                    # CHANGED (Claude): both lines are now wrapped in np.sqrt(). They compute
                    # "mean of the variances + variance of the means", which is a variance, but
                    # are stored in the StdDev_* columns and consumed as sigmas (plt_satmag
                    # passes them to errorbar's yerr). The matching per-point calculation
                    # further down already applied .pow(0.5); these did not, so the averaged
                    # temperature/field error bars were previously variances. Note this shrinks
                    # reported deviations below 1 and grows those above 1 relative to the old
                    # values, so averaged error bars will not match earlier runs.
                    new_row[self.labels["TEMP_DEV"]] = np.sqrt(
                        np.divide(
                            tf_df[self.labels["TEMP_DEV"]]
                            .pow(2)
                            .mul(tf_df["Num_Temp_Checks"].mean())
                            .sum(),
                            (tf_df["Num_Temp_Checks"].mean() * len(tf_df)),
                        )
                        + np.divide(
                            tf_df[self.labels["TEMP_AVG"]]
                            .sub(tf_df[self.labels["TEMP_AVG"]].mean())
                            .pow(2)
                            .mul(tf_df["n_replicates"].mean())
                            .sum(),
                            (tf_df["n_replicates"].mean() * len(tf_df)),
                        )
                    )
                    new_row[self.labels["FIELD_DEV"]] = np.sqrt(
                        np.divide(
                            tf_df[self.labels["FIELD_DEV"]]
                            .pow(2)
                            .mul(tf_df["Num_Temp_Checks"].mean())
                            .sum(),
                            (tf_df["Num_Temp_Checks"].mean() * len(tf_df)),
                        )
                        + np.divide(
                            tf_df[self.labels["FIELD_AVG"]]
                            .sub(tf_df[self.labels["FIELD_AVG"]].mean())
                            .pow(2)
                            .mul(tf_df["n_replicates"].mean())
                            .sum(),
                            (tf_df["n_replicates"].mean() * len(tf_df)),
                        )
                    )
                    new_row["averaged"] = ",".join(tf_df["id"].to_list())
                    new_row["id"] = f"{t}K_{f}T"
                    nScansPerPoint = tf_df["n_replicates"].mean()
                    new_row["n_replicates"] = nScansPerPoint * len(tf_df)
                    new_row["Num_Temp_Checks"] = tf_df["Num_Temp_Checks"].mean() * len(
                        tf_df
                    )
                    # Update the data
                    first_data = self.info_df["data"].iloc[0]
                    cols_to_avg = [
                        col for col in list(first_data.columns) if "_avg" in col
                    ]
                    cols_to_std = [
                        col for col in list(first_data.columns) if "_std" in col
                    ]
                    # Make a new avg column
                    new_datadf = pd.DataFrame()
                    # Check that NM cols match of if not align the dfs
                    nm_data = pd.DataFrame(
                        [
                            tf_df.at[i, "data"][self.labels["NM"]]
                            for i in range(len(tf_df))
                        ]
                    )
                    if (
                        not nm_data.mean(axis=0)
                        .round(1)
                        .equals(tf_df.at[0, "data"][self.labels["NM"]].round(1))
                    ):
                        # NOTE: `first` (row 0 re-aligned onto the union index) is discarded, so if
                        # a later replicate contributes wavelengths row 0 lacks, row 0 keeps its
                        # shorter index and the column-wise averaging below will misalign.
                        for i in range(len(tf_df)):
                            tf_df.at[i, "data"].set_index(
                                self.labels["NM"], inplace=True, drop=True
                            )
                            first, new = tf_df.at[0, "data"].align(
                                tf_df.at[i, "data"], join="outer", axis=0
                            )
                            tf_df.at[i, "data"] = new
                    for col in first_data.columns:
                        col_data = pd.DataFrame(
                            [tf_df.at[i, "data"][col] for i in range(len(tf_df))]
                        )
                        if col in cols_to_avg:
                            new_datadf[col] = col_data.mean(axis=0)
                        elif col in cols_to_std:
                            # new_datadf[col] = col_data.pow(2).sum(axis=0).pow(0.5).mul(0.5)

                            # Mean of the variances + Variance of the means (https://arxiv.org/pdf/1007.1012)
                            avgcol = pd.DataFrame(
                                [
                                    tf_df.at[i, "data"][col.replace("_std", "_avg")]
                                    for i in range(len(tf_df))
                                ]
                            )
                            new_datadf[col] = col_data.pow(2).mul(nScansPerPoint).sum(
                                axis=0
                            ).div(nScansPerPoint * len(tf_df)) + avgcol.sub(
                                avgcol.mean(axis=0)
                            ).pow(
                                2
                            ).mul(
                                nScansPerPoint
                            ).sum(
                                axis=0
                            ).div(
                                nScansPerPoint * len(tf_df)
                            )
                            new_datadf[col] = new_datadf[col].pow(0.5)
                        else:
                            if (
                                not col_data.mean(axis=0)
                                .round(1)
                                .equals(tf_df.at[0, "data"][col].round(1))
                            ):
                                print(
                                    f"Warning: Values in {col} are not all the same. Averaging anyways."
                                )
                                print(col_data.head(2))
                                print(col_data.tail(1))
                            new_datadf[col] = col_data.mean(axis=0)
                    new_datadf.reset_index(inplace=True)
                    new_data_dfs.append(new_datadf)
                    # print(new_row)
                    new_info_df = pd.concat(
                        [new_info_df, pd.Series(new_row).to_frame().T],
                        ignore_index=True,
                    )
        # Write to new MCD data structure
        new_info_df["data"] = new_data_dfs
        averaged_data = self.copy()
        averaged_data.info_df = new_info_df
        return averaged_data


#: Backwards-compatible aliases for the old class names used in existing notebooks.
MCD_Data = MCD
VTVH_MCD_Data = VTVH_MCD


# ---------------------------------------------------------------------------
# Code written by RG (Robert Gipson).
# Claude (Opus 5) reviewed and adjusted this file:
#   - Fixed the imports, which broke the module three ways: `from abscd import *`
#     was an absolute import that fails inside a package, the parent class
#     `AbsCD_Data` did not exist under that name, and math/os/numpy/pandas/scipy
#     were used without being imported.
#   - Renamed MCD_Data -> MCD and VTVH_MCD_Data -> VTVH_MCD to match the package's
#     other classes (aliases kept above), and replaced the string parameter
#     annotations on both __init__s with real type hints.
#   - subtract(): sub_id now resets per zero-field scan (a zero that matched
#     nothing re-subtracted the previous zero's list), and next_zero/prev_zero are
#     bounds-checked in the same_temp=False branch, where zi+1 ran off the end of
#     zero_idx and zi-1 wrapped to the last element.
#   - plt_satmag(): `colors.extend(colors)` was mutating the default argument, so
#     the default palette doubled in length on every call that needed extra
#     colors; it now cycles into a local list. Same for the `fontdict={}` default.
#     Shared palettes moved to a module-level PLT_COLORS tuple.
#   - add_satmag_y(): writes to inten_x_std positionally instead of by info_df
#     label, hoists the ystd_col default out of the loop, and computes the
#     nearest-point lookup once per row rather than three times.
#   - resid_doublet_vtvh() / fit_doublet_model_vtvh(): converts the field/temp/
#     intensity Series to arrays before positional indexing, and raises instead of
#     returning None on a length mismatch (which crashed least_squares).
#   - VTVH_MCD.load(): the `channel_df.drop(...)` dropping the duplicate x column
#     was a no-op (no inplace, no assignment); os.path.join for paths; npts resets
#     per file; both load()s now accept **kwargs and return True.
#   - quick_plot() passes through to the parent by keyword rather than by position.
#   - loc_field() copies its slice; subtract_component() skips the deep copy when
#     inplace=True; replaced `self.info_df.at[0,'data']` with `['data'].iloc[0]`.
#   - Corrected three numerical issues, each marked "CHANGED (Claude)" at the line:
#     subtract_component now scales the component variance by scalar**2 rather than
#     scalar; average_replicates square-roots the averaged TEMP_DEV/FIELD_DEV so the
#     StdDev_* columns hold sigmas rather than variances; and resid_doublet_vtvh
#     returns the per-point signed residual vector least_squares expects instead of
#     one summed absolute value. All three change fitted values or error bars, so
#     results will not match earlier runs.
#   - Added a NOTE (no change) on the discarded `first` frame in the replicate
#     alignment loop of average_replicates.
# ---------------------------------------------------------------------------
