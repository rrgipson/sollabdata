"""Tests for the DFT class (sollabdata/dft.py).

Run the whole suite from the repo root:

    pytest

Run just this file, or just one test, showing print output:

    pytest tests/test_dft.py
    pytest tests/test_dft.py::test_loads_one_row_per_json_file -s

Fixture data lives in tests/data/dft/. h2o_only_data.json is a cclib-parsed
Gaussian calculation on water: a geometry optimization, a single point, and a
frequency job, in that order.
"""

from pathlib import Path

import pytest

from sollabdata import DFT

# Path to the fixture folder, resolved relative to THIS file rather than to the
# current working directory -- so the tests pass no matter where pytest is run from.
# __file__ is tests/test_dft.py, so .parent is tests/.
DATA_DIR = Path(__file__).parent / "data" / "dft"


@pytest.fixture
def water():
    """A DFT object loaded from the water calculation in tests/data/dft/.

    A pytest "fixture" is a named setup function. Any test that takes an argument
    called `water` gets the return value of this function passed in, and pytest
    re-runs it fresh for each test -- so one test mutating the object cannot affect
    another. This is the main reason to use a fixture rather than a module-level
    global.
    """
    # DFT() with path_to_raw_data and no info_csv scans the folder for .json files
    # and builds a generic info_df, using each file's stem as its id.
    return DFT(path_to_raw_data=str(DATA_DIR))


def test_loads_one_row_per_json_file(water):
    """Reading the fixture folder gives one info_df row, with the parsed table attached.

    The name matters: pytest collects functions starting with `test_`, and the name
    is what shows up on failure, so it should read as the claim being made.
    """
    # ARRANGE -- the `water` fixture already did this for us.

    # ACT / ASSERT. A plain `assert` is the whole API; pytest rewrites it so a
    # failure prints both sides of the comparison.
    assert len(water.info_df) == 1

    # The id defaults to the file name without its extension.
    assert water.info_df.at[0, "id"] == "h2o_only_data"
    assert water.info_df.at[0, "File"] == "h2o_only_data.json"

    # The `data` column holds a nested DataFrame -- one row per job stage
    # (opt, sp, freq) and one column per cclib attribute.
    data = water.info_df.at[0, "data"]
    assert data.shape[0] == 3

    # Spot-check that real cclib attributes came through rather than an empty frame.
    for attribute in ("atomnos", "atomcoords", "scfenergies", "vibfreqs"):
        assert attribute in data.columns


# ---------------------------------------------------------------------------
# Written by RG (Robert Gipson). Created by Claude (Opus 5): first test module for
# the DFT class -- a DATA_DIR path anchored to __file__, a `water` fixture that
# loads tests/data/dft/, and one test asserting that reading the folder produces a
# single info_df row with the id/File columns and a 3-stage nested data table.
# ---------------------------------------------------------------------------
