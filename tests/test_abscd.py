"""Tests for the AbsCD class (sollabdata/abscd.py).

Run the whole suite from the repo root:

    pytest

Fixture data lives in tests/data/abscd/.
"""

from pathlib import Path

import pytest

from sollabdata import AbsCD

# Path to the fixture folder, resolved relative to THIS file rather than to the
# current working directory -- so the tests pass no matter where pytest is run from.
# __file__ is tests/test_dft.py, so .parent is tests/.
DATA_DIR = Path(__file__).parent / "data" / "abscd"


@pytest.fixture
def abscd_spectra():
    """Two AbsCD spectra loaded from the J-1700 scans in tests/data/abscd/.

    The first scan is a positive gaussian feature at 600nm (inten +1 mdeg) with a width of 100nm, 
    and the second scan is a negative feature (-1 mdeg) at 800nm with a width of 200nm.
    """
    # DFT() with path_to_raw_data and no info_csv scans the folder for .json files
    # and builds a generic info_df, using each file's stem as its id.
    return AbsCD(path_to_raw_data=str(DATA_DIR), info_csv=str(DATA_DIR)+"/abscd_test_info.csv")


def test_loads_one_row_per_scan(abscd_spectra):
    # ARRANGE -- the fixture already did this for us.

    # ACT / ASSERT. A plain `assert` is the whole API; pytest rewrites it so a
    # failure prints both sides of the comparison.
    assert len(abscd_spectra.info_df) == 2

    # The id and filename are read from the info_csv.
    assert abscd_spectra.info_df.at[0, "id"] == "Positive"
    assert abscd_spectra.info_df.at[1, "id"] == "Negative"
    assert abscd_spectra.info_df.at[0, "File"] == "01_pos-1.csv"

    # The `data` column holds a nested DataFrame -- this data should 
    # be formatted with npts from .csv as number of rows and 4 columns
    data = abscd_spectra.info_df.at[0, "data"]
    assert data.shape == (1591, 5)



# ---------------------------------------------------------------------------
# Written by RG (Robert Gipson). Template created by Claude (Opus 5).
# ---------------------------------------------------------------------------
