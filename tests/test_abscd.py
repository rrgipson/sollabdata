"""Tests for the AbsCD class (sollabdata/abscd.py).

Run the whole suite from the repo root:

    pytest

Fixture data lives in tests/data/abscd/: two real J-1700 exports plus the info csv
that keys them.
"""

from pathlib import Path

import pytest

from sollabdata import AbsCD

# Path to the fixture folder, resolved relative to THIS file rather than to the
# current working directory -- so the tests pass no matter where pytest is run from.
# __file__ is tests/test_abscd.py, so .parent is tests/.
DATA_DIR = Path(__file__).parent / "data" / "abscd"

# The channels the J-1700 wrote, in column order. The x label comes from the XUNITS
# header line and the y labels from YUNITS/Y2UNITS/Y3UNITS/Y4UNITS.
EXPECTED_COLUMNS = ["NANOMETERS", "CD/DC [mdeg]", "DC [V]", "HT [V]", "ABSORBANCE"]

# The CD channel name, used by most of the fitting tests below.
CD_COL = "CD/DC [mdeg]"


@pytest.fixture
def abscd_spectra():
    """Two AbsCD spectra loaded from the J-1700 scans in tests/data/abscd/.

    The first scan is a positive gaussian feature at 600nm (inten +1 mdeg) with a width of 100nm,
    and the second scan is a negative feature (-1 mdeg) at 800nm with a width of 200nm.
    Both also carry an ABSORBANCE channel peaking at 1.0 at the same wavelength.

    A pytest "fixture" is a named setup function: any test taking an argument called
    `abscd_spectra` gets this return value, rebuilt fresh for that test. That
    isolation matters here because most AbsCD methods mutate info_df in place.
    """
    # A bare info_csv file name is resolved against path_to_raw_data, and is excluded
    # from the data files that get loaded.
    return AbsCD(path_to_raw_data=str(DATA_DIR), info_csv="abscd_test_info.csv")


def test_loads_one_row_per_scan(abscd_spectra):
    # ARRANGE -- the fixture already did this for us.

    # ACT / ASSERT. A plain `assert` is the whole API; pytest rewrites it so a
    # failure prints both sides of the comparison.
    assert len(abscd_spectra.info_df) == 2

    # The id and filename are read from the info_csv.
    assert abscd_spectra.info_df.at[0, "id"] == "Positive"
    assert abscd_spectra.info_df.at[1, "id"] == "Negative"
    assert abscd_spectra.info_df.at[0, "File"] == "01_pos-1.csv"


def test_loads_data_from_j1700(abscd_spectra):
    # The `data` column holds a nested DataFrame -- NPOINTS from the .csv header as
    # the number of rows, and one column per channel (1 x + 4 y = 5).
    data = abscd_spectra.info_df.at[0, "data"]
    assert data.shape == (1591, 5)

    # Assert the names too, not just the count: they are derived by string-matching
    # the header lines, which is the fragile part of the parser.
    assert list(data.columns) == EXPECTED_COLUMNS


def test_fit_gaussians(abscd_spectra):
    # Test fitting procedure. fit_gaussians returns (results, details, fit):
    # the raw parameter array, a tidy per-band DataFrame, and the scipy result.
    # Underscore-prefixed names mark the two we do not assert on here.
    _results, details, _fit = abscd_spectra.fit_gaussians(
        energies=[700],
        fwhm=[150],
        intens=[1.5],
        id="Positive",
        x_col="NANOMETERS",
        y_cols="CD/DC [mdeg]",
    )
    # Assert on the named `details` columns rather than positions in `results`, so
    # these do not silently break if the fitvars packing order ever changes.
    # Test correctly fitted center
    assert details.at[0, "Energy"] == pytest.approx(600, abs=0.1)
    # Test correctly fitted fwhm
    assert details.at[0, "FWHM"] == pytest.approx(100, abs=0.1)
    # Test correctly fitted intensity
    assert details.at[0, "Inten_y0"] == pytest.approx(1, abs=0.1)


def test_fit_gaussians_negative_feature(abscd_spectra):
    """The second scan's band is negative, which is a different path through resid()."""
    _results, details, _fit = abscd_spectra.fit_gaussians(
        energies=[750],
        fwhm=[250],
        intens=[-1.5],
        id="Negative",
        x_col="NANOMETERS",
        y_cols="CD/DC [mdeg]",
    )
    assert details.at[0, "Energy"] == pytest.approx(800, abs=0.1)
    assert details.at[0, "FWHM"] == pytest.approx(200, abs=0.1)
    assert details.at[0, "Inten_y0"] == pytest.approx(-1, abs=0.1)


def test_fit_gaussians_accepts_non_binding_bounds(abscd_spectra):
    """Passing bounds must not raise, and slack bounds must not distort the answer.

    The J-1700 writes x descending (DELTAX,-1), so the area integral comes back
    negative. Dividing low_bds/up_bds by a negative area swaps them, and
    least_squares then rejected the bounds outright with "Each lower bound must be
    strictly less than each upper bound".

    The true values sit well inside this box, so this test says nothing about
    whether bounds are ENFORCED -- that is what the clamping test below is for.
    What it does pin is that supplying slack bounds still recovers the same
    parameters as the unconstrained fit above.
    """
    _results, details, _fit = abscd_spectra.fit_gaussians(
        energies=[700],
        fwhm=[150],
        intens=[1.5],
        id="Positive",
        x_col="NANOMETERS",
        y_cols="CD/DC [mdeg]",
        low_bds=[500, 50, 0.1],
        up_bds=[700, 200, 5.0],
    )
    assert details.at[0, "Energy"] == pytest.approx(600, abs=0.1)
    assert details.at[0, "FWHM"] == pytest.approx(100, abs=0.1)
    assert details.at[0, "Inten_y0"] == pytest.approx(1, abs=0.1)


@pytest.mark.parametrize(
    ("guess", "low_bds", "up_bds", "column", "at_the_wall"),
    [
        # Inten_y0 is the slice fit_gaussians area-normalizes internally, so these
        # two cases are the ones that genuinely exercise that normalization.
        pytest.param(
            ([600], [100], [0.3]), [400, 50, 0.1], [800, 200, 0.5], "Inten_y0", 0.5,
            id="intensity_ceiling",
        ),
        pytest.param(
            ([600], [100], [3.0]), [400, 50, 2.0], [800, 200, 5.0], "Inten_y0", 2.0,
            id="intensity_floor",
        ),
        # Energy and FWHM bounds pass through unnormalized; included as a control.
        pytest.param(
            ([500], [100], [0.9]), [400, 50, 0.1], [550, 200, 5.0], "Energy", 550.0,
            id="energy_ceiling",
        ),
    ],
)
def test_fit_gaussians_clamps_to_bounds(
    abscd_spectra, guess, low_bds, up_bds, column, at_the_wall
):
    """A bound that excludes the true value must stop the fit at the wall.

    Truth for this scan is Energy=600, FWHM=100, Inten_y0=1. Each case boxes the
    true value out, so a fit that quietly discarded its bounds would sail past to
    the truth and fail the assertion -- which is what makes this test, rather than
    the slack-bounds one above, the evidence that bounds are enforced.

    Note the initial guess has to stay INSIDE the box: least_squares rejects an
    out-of-bounds guess with "Initial guess is outside of provided bounds" before
    it ever starts iterating.

    `@pytest.mark.parametrize` runs the body once per case, reporting each under
    its own `id` so a single failing case is named in the output.
    """
    energies, fwhm, intens = guess
    _results, details, _fit = abscd_spectra.fit_gaussians(
        energies=energies,
        fwhm=fwhm,
        intens=intens,
        id="Positive",
        x_col="NANOMETERS",
        y_cols="CD/DC [mdeg]",
        low_bds=low_bds,
        up_bds=up_bds,
    )
    assert details.at[0, column] == pytest.approx(at_the_wall, rel=1e-4)


def test_channel_labels_come_from_anchored_header_keys(tmp_path):
    """A y channel whose unit string contains an "X" is still a y channel.

    `tmp_path` is a builtin pytest fixture: a fresh empty directory per test,
    cleaned up afterwards. Handy for writing a tiny synthetic input rather than
    committing another fixture file.

    The header keys are XUNITS then YUNITS, Y2UNITS, Y3UNITS, ... -- so the parser
    matches on those anchored keys. Testing against "FLUX" because the earlier
    `"UNITS" in line and "X" in line` check classified it as the x axis, which
    silently shifted every column name over by one.
    """
    scan = tmp_path / "flux_scan.csv"
    scan.write_text(
        "TITLE,synthetic\n"
        "XUNITS,NANOMETERS\n"
        "YUNITS,FLUX\n"          # contains an X, but is a y channel
        "Y2UNITS,ABSORBANCE\n"
        "NPOINTS,3\n"
        "XYDATA\n"
        "600,1.0,0.5\n"
        "599,2.0,0.6\n"
        "598,3.0,0.7\n"
    )

    spectra = AbsCD(path_to_raw_data=str(tmp_path))
    data = spectra.info_df.at[0, "data"]

    assert list(data.columns) == ["NANOMETERS", "FLUX", "ABSORBANCE"]
    assert data.shape == (3, 3)
    # and the values landed under the right headings
    assert data["FLUX"].tolist() == [1.0, 2.0, 3.0]
    assert data["ABSORBANCE"].tolist() == [0.5, 0.6, 0.7]


def test_add_eps_scales_by_concentration(abscd_spectra):
    """eps = A / (c * l), with c read per-row from the info_csv Conc column."""
    abscd_spectra.add_eps("Conc", conc_units="mM", path_length=1)

    # Each scan's ABSORBANCE peaks at 1.0 at its own feature wavelength, so:
    #   scan 1: A=1.0 at 600nm, c=1.0mM -> eps = 1.0/1e-3 = 1000
    #   scan 2: A=1.0 at 800nm, c=2.0mM -> eps = 1.0/2e-3 =  500
    # The two rows having DIFFERENT concentrations is the point: a bug that reached
    # for the wrong row's Conc would show up here and nowhere else.
    for idx, peak_nm, expected in [(0, 600.0, 1000.0), (1, 800.0, 500.0)]:
        data = abscd_spectra.info_df.at[idx, "data"]
        at_peak = data.loc[data["NANOMETERS"] == peak_nm]
        assert float(at_peak["eps"].iloc[0]) == pytest.approx(expected)


def test_fit_gaussians_multiple_ids_share_bands(abscd_spectra):
    """Two spectra fit at once share energies and FWHMs but keep separate intensities.

    The fixtures hold bands in different places -- Positive is +1 mdeg at 600 nm
    (100 nm FWHM), Negative is -1 mdeg at 800 nm (200 nm FWHM). Fitting both against
    the same two-band model should recover both band positions, with each trace's
    intensities selecting only its own band. That is the whole point of a shared fit:
    one set of band parameters, per-spectrum amplitudes.
    """
    _results, details, _fit = abscd_spectra.fit_gaussians(
        energies=[610, 790],
        fwhm=[110, 190],
        # num_gauss per trace, id-major: Positive's two bands, then Negative's two
        intens=[0.9, 0.1, 0.1, -0.9],
        id=["Positive", "Negative"],
        x_col="NANOMETERS",
        y_cols=CD_COL,
    )

    # shared parameters
    assert details["Energy"].tolist() == pytest.approx([600, 800], abs=0.5)
    assert details["FWHM"].tolist() == pytest.approx([100, 200], abs=0.5)
    # Inten_y0 is Positive/CD, Inten_y1 is Negative/CD
    assert details["Inten_y0"].tolist() == pytest.approx([1, 0], abs=1e-3)
    assert details["Inten_y1"].tolist() == pytest.approx([0, -1], abs=1e-3)


def test_fit_gaussians_multiple_ids_store_per_id_slices(abscd_spectra):
    """Each id gets a self-contained fit slice, so check_plot(id) still works.

    A multi-id fit has one intensity block per trace, but the slice stored on a given
    row holds only the shared energies/widths plus that id's own block -- the same
    layout a single-id fit produces.
    """
    abscd_spectra.fit_gaussians(
        energies=[610, 790], fwhm=[110, 190], intens=[0.9, 0.1, 0.1, -0.9],
        id=["Positive", "Negative"], x_col="NANOMETERS", y_cols=CD_COL,
    )

    for one_id, expected_inten in [("Positive", 1.0), ("Negative", -1.0)]:
        idx = abscd_spectra.info_df.index[abscd_spectra.info_df["id"] == one_id][0]
        stored = abscd_spectra.info_df.at[idx, "fit"]
        # 2 bands x (2 shared blocks + 1 y column) = 6, not the full 8 of the fit
        assert len(stored) == 6
        assert stored[:2] == pytest.approx([600, 800], abs=0.5)

    # and check_plot can still infer num_gauss from one id's slice
    fig = abscd_spectra.check_plot("Negative", x_col="NANOMETERS", y_cols=CD_COL)
    assert len(fig.data) == 4  # data + total fit + 2 component gaussians


def test_fit_gaussians_crosses_ids_with_y_cols(abscd_spectra):
    """ids and y_cols multiply: 2 ids x 2 y columns is 4 traces, ordered id-major."""
    _results, details, _fit = abscd_spectra.fit_gaussians(
        energies=[610, 790], fwhm=[110, 190],
        # Positive/CD, Positive/Abs, Negative/CD, Negative/Abs
        intens=[0.9, 0.1, 0.9, 0.1, 0.1, -0.9, 0.1, 0.9],
        id=["Positive", "Negative"], x_col="NANOMETERS", y_cols=[CD_COL, "ABSORBANCE"],
    )

    assert [c for c in details.columns if c.startswith("Inten")] == [
        "Inten_y0", "Inten_y1", "Inten_y2", "Inten_y3"
    ]
    # Negative's absorbance is positive (+1) even though its CD is negative (-1)
    assert details["Inten_y2"].tolist() == pytest.approx([0, -1], abs=1e-3)
    assert details["Inten_y3"].tolist() == pytest.approx([0, 1], abs=1e-3)


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        pytest.param(dict(intens=[0.9, 0.1]), "intens must hold num_gauss per trace",
                     id="intens_too_short"),
        pytest.param(dict(low_bds=[0, 0, 0], up_bds=[1, 1, 1]),
                     "must cover energies, widths and every trace", id="bounds_wrong_length"),
        pytest.param(dict(scalar=[1.0]), "scalar must hold one value per trace",
                     id="scalar_wrong_length"),
        pytest.param(dict(id=["Positive", "Nope"]), "No info_df row with id",
                     id="unknown_id"),
        pytest.param(dict(fwhm=[110]), "energies and fwhm must be the same length",
                     id="fwhm_length_mismatch"),
    ],
)
def test_fit_gaussians_rejects_inconsistent_inputs(abscd_spectra, kwargs, expected_message):
    """Wrong-length inputs raise a message naming the expected size.

    Without these checks a mis-sized `intens` either silently mis-slices the
    parameter vector or surfaces as an opaque error from deep inside scipy.
    """
    call = dict(
        energies=[610, 790], fwhm=[110, 190], intens=[0.9, 0.1, 0.1, -0.9],
        id=["Positive", "Negative"], x_col="NANOMETERS", y_cols=CD_COL,
    )
    call.update(kwargs)
    with pytest.raises(ValueError, match=expected_message):
        abscd_spectra.fit_gaussians(**call)


def test_fit_gaussians_requires_a_common_x_grid(tmp_path):
    """same_x=True refuses spectra on different x grids; same_x=False allows them."""
    import numpy as np

    def write(name, lo, hi):
        xs = np.arange(hi, lo - 1, -1.0)
        body = "".join(f"{x},{np.exp(-0.5 * ((x - 600) / 50) ** 2)}\n" for x in xs)
        (tmp_path / name).write_text(
            f"XUNITS,NANOMETERS\nYUNITS,{CD_COL}\nNPOINTS,{len(xs)}\nXYDATA\n" + body
        )

    write("wide.csv", 300, 900)
    write("narrow.csv", 400, 800)
    (tmp_path / "key.csv").write_text("id,File\nwide,wide.csv\nnarrow,narrow.csv\n")

    spectra = AbsCD(path_to_raw_data=str(tmp_path), info_csv="key.csv")
    shared = dict(energies=[610], fwhm=[60], intens=[1.0, 1.0],
                  id=["wide", "narrow"], x_col="NANOMETERS", y_cols=CD_COL)

    with pytest.raises(ValueError, match="not on the same NANOMETERS grid"):
        spectra.fit_gaussians(**shared)

    # with the guard lifted, each trace is integrated and fit against its own grid
    _results, details, _fit = AbsCD(
        path_to_raw_data=str(tmp_path), info_csv="key.csv"
    ).fit_gaussians(**shared, same_x=False)
    assert details.at[0, "Energy"] == pytest.approx(600, abs=0.5)
    # sigma=50 -> FWHM = 50 * 2*sqrt(2*ln2)
    assert details.at[0, "FWHM"] == pytest.approx(117.74, abs=0.5)


# ---------------------------------------------------------------------------
# Written by RG (Robert Gipson). Template created by Claude (Opus 5), which also
# made these adjustments: corrected the stale test_dft.py/DFT comments, switched the
# fixture to a bare info_csv name (exercising path resolution), asserted the parsed
# column names alongside the shape, moved the fit assertions onto the named `details`
# columns, and added the negative-feature, bounded-fit, add_eps, and anchored
# header-key tests.
# ---------------------------------------------------------------------------
