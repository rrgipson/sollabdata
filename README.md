# sollabdata

Python tools for reading, processing, and plotting Solomon Lab instrument data — currently 
Abs/CD and MCD spectra from the J-1700, and DFT calculation output — with one consistent
interface across instruments. This python package was adapted with permission by Robert Gipson 
(with the assistance of Claude Code Opus 5) from code developed by Eleanor Dunietz (edx4900) 
and Robert Gipson (rrgipson), specifically from the analysis module of the 
electronic lab notebook setup found here: https://github.com/edx4900/lab-notebook--jupyter.


> **Status:** personal/lab project, pre-1.0. The API is still under development and method
> signatures may change without notice. Not on PyPI; install from a clone.

## Install

```bash
git clone https://github.com/rrgipson/sollabdata.git
cd sollabdata
pip install -e ".[dev]"
```

The `-e` (editable) install means edits to the source take effect immediately, with no
reinstall — which is what you want while the package is still changing. The `[dev]`
extra adds `pytest` so you can run the test suite; use `pip install -e .` if you only
want to use the package. There is also a `[notebook]` extra (`ipython`) — see
[Use in a Jupyter notebook](#use-in-a-jupyter-notebook). Jupyter itself is not listed,
since it is usually already installed and is better installed by conda than pulled in
by pip alongside a conda environment.

If you are starting from a fresh conda environment:

```bash
conda create -n sollabdata python=3.12
conda activate sollabdata
pip install -e ".[dev]"
```

### Dependencies

These are declared in `pyproject.toml` and installed automatically, but listed here so
you know what is in play and why:

| Package | Minimum | Used for |
|---|---|---|
| `python` | 3.9 | — |
| `pandas` | 1.3 | the `info_df` data model; all csv/json reading |
| `numpy` | 1.20 | array math throughout |
| `scipy` | 1.7 | `least_squares` for band and VTVH fitting, `quad` for the doublet model |
| `plotly` | 5.0 | interactive plots (`quick_plot`, `check_plot`) |
| `matplotlib` | 3.4 | static/publication plots (`plt_satmag`, and the snippets `prep_plt` emits) |
| `tabulate` | 0.8.9 | `info_df.to_markdown()`, used by `to_md()` |
| `cclib` | 1.7 | unit conversion (`convertor`) in the `DFT` class |
| `pytest` | 7.0 | test suite only (`[dev]` extra) |

Note that `cclib` is used here only to *read* calculation output. Producing the
per-calculation json files from raw ORCA/Gaussian output is a separate upstream step
that `sollabdata` does not perform.

## Quickstart

Runnable from the repo root against the test fixtures, so you can try it before
pointing it at your own data:

```python
from sollabdata import AbsCD

# every row of info_df is one scan; its nested "data" column holds that spectrum
spectra = AbsCD(
    path_to_raw_data="tests/data/abscd/",
    info_csv="abscd_test_info.csv",
)

print(spectra.info_df.drop(columns="data"))
#    ScanNum  Equiv        id  Conc  Temp(C)          File
# 0        1    0.0  Positive   1.0      4.0  01_pos-1.csv
# 1        2    1.0  Negative   2.0      4.0  02_neg-1.csv

print(spectra.info_df["data"].iloc[0].columns.tolist())
# ['NANOMETERS', 'CD/DC [mdeg]', 'DC [V]', 'HT [V]', 'ABSORBANCE']

# derived axes and units are added to every scan's nested frame
spectra.add_wavenums()                                    # -> "Wavenums" column
spectra.add_eps("Conc", conc_units="mM", path_length=1)   # -> "eps" column

# fit the 600 nm CD band
_results, details, _fit = spectra.fit_gaussians(
    energies=[700], fwhm=[150], intens=[1.5],     # initial guesses
    id="Positive", x_col="NANOMETERS", y_cols="CD/DC [mdeg]",
)
print(details.round(2).to_string(index=False))
#  Energy  FWHM  Inten_y0   f
#   600.0 100.0       1.0 0.0
```

`add_eps` takes `"Conc"` as a *column name* in `info_df`, so each scan is scaled by its
own concentration. Passing a number instead applies that one value to every scan.

## Use in a Jupyter notebook

The package is written for interactive analysis in a notebook rather than for scripted
pipelines, and several pieces only make sense in that setting:

- **`quick_plot()` and `check_plot()` return plotly figures**, which render inline and
  stay interactive — hover readouts, zoom, and clicking the legend to toggle a scan.
  Because traces for one `id` share a `legendgroup`, one legend click hides that scan in
  every stacked subplot at once. Outside a notebook you need an explicit `fig.show()`.

- **`to_md()` returns markdown, meant to be wrapped for display:**

  ```python
  from IPython.display import Markdown as md

  md(data.to_md())        # info_df as a rendered table, minus the nested data column
  ```

  This is the one place the package genuinely needs IPython, and the reason for the
  `[notebook]` extra.

- **`prep_plt()` and `prep_animation_plt()` print matplotlib source code rather than
  drawing anything.** Run one, then paste the output into the next cell and edit it by
  hand — the intended workflow when you need a publication figure with axis limits,
  tick formatting, and labels tuned per plot. `prep_animation_plt()` emits a loop that
  draws one figure per scan with earlier traces faded behind, for stepping through a
  series.

- **Methods print progress as they work** — the `Parsing Data from ...` line, fit
  convergence reports, and which scan was subtracted from which. That is intentional
  feedback for cell-by-cell work, not stray debugging output. Pass `-s` to pytest if
  you want to see it during tests.

- **In-place mutation suits cell-by-cell work, but watch out for re-running a cell.**
  Methods that write a derived column (`add_wavenums`, `add_eps`) just overwrite it, and
  re-running them is harmless. Methods that *combine* two spectra are the ones to be
  careful with: `add()` accumulates, so running its cell twice doubles the contribution
  it added. Keep a pristine object with `obj.copy()` — see
  [the data model](#the-data-model) — so you can recover without re-parsing from disk.
  Re-running the constructor cell is the other clean reset.

## The data model

This is the one concept worth understanding before anything else — everything in the
package is built on it.

Each object holds a single **`info_df`** DataFrame with **one row per measurement**:

- **flat columns** are metadata for that measurement — `id`, `File`, `ScanNum`,
  concentration, field, temperature, whatever your info csv carries
- the **`data` column** is nested: each cell holds a whole DataFrame of that
  measurement's x/y traces

```
info_df
┌────┬───────────┬──────────────┬──────┬─────────────────────────────────┐
│ id │ Scan_Num  │ Expt_Param   │ ...  │ data                            │
├────┼───────────┼──────────────┼──────┼─────────────────────────────────┤
│ s1 │ 1         │ 0.0          │ ...  │ DataFrame: nm, CD, Abs, ...     │
│ s2 │ 2         │ 7.0          │ ...  │ DataFrame: nm, CD, Abs, ...     │
└────┴───────────┴──────────────┴──────┴─────────────────────────────────┘
```

Consequences worth internalizing:

- **A spectrum is reached in two steps:** `obj.info_df.at[0, "data"]["NANOMETERS"]`.
- **Most methods mutate in place** and return `True`/`None` rather than a new object.
  Use `obj.copy()` — which deep-copies the nested frames — before a destructive step
  you might want to undo.
- **Per-scan operations are broadcast.** `add_wavenums()`, `subtract()`, and
  `baseline()` loop over every row for you; you rarely index a single spectrum yourself.
- **`labels` maps short names to column headers.** `MCD` and `VTVH_MCD` populate it, so
  `obj.labels["CD"]` gives the right CD column for that instrument mode rather than you
  hard-coding `"CD/DC [mdeg]"` vs `"CD/DC [mdeg]_avg"`.

## Classes

```
LabData             base: info_df bookkeeping, copy/drop, read/write, plotting
├── AbsCD           J-1700 Abs/CD parsing, spectral arithmetic, Gaussian band fitting
│   └── MCD         field/temperature handling, 0 T subtraction, saturation magnetization
│       └── VTVH_MCD    interval-scan mode with replicate averaging
└── DFT             cclib-parsed calculation output
```

```python
from sollabdata import LabData, AbsCD, MCD, VTVH_MCD, DFT
```

All five take the same constructor arguments: `path_to_raw_data`, `info_csv`, `info_df`,
`experiment_df`, `processing_metadata`, `labels`. Passing `path_to_raw_data` parses the
folder immediately; `info_csv` may be a bare file name in that folder or a full path.

### LabData — shared machinery

| Method | Does |
|---|---|
| `copy()` | deep copy, including the nested `data` frames |
| `drop(ids=, idx=)` | remove measurements by id or index |
| `read()` / `load()` / `process()` | the parsing pipeline; children override `load` (and `read` for DFT) |
| `write(path)` | save each spectrum plus an `info.csv` to a new folder |
| `quick_plot(x=, y=)` | interactive plotly figure, one subplot row per y column |
| `prep_plt()` / `prep_animation_plt()` | print matplotlib source to paste into a notebook and hand-edit |
| `to_md()` | `info_df` as markdown, for `IPython.display.Markdown` |
| `get_idx_from_id(ids)` | index labels for the given ids |
| `apply_by_row(func, ...)` | run a per-id function over every row, collecting results into `info_df` |

### AbsCD — J-1700 Abs/CD

| Method | Does |
|---|---|
| `subtract(ref_id, ys)` / `add(...)` | subtract or add one spectrum from/to the others (`square=True` adds in quadrature, for error propagation) |
| `baseline(ys, x_range)` | zero a featureless region |
| `fix_changeover(ys, x_change)` | correct the detector-changeover discontinuity |
| `add_wavenums()` / `add_eV()` | derived x axes from nanometers |
| `add_eps(conc)` / `add_deps(conc)` | Beer–Lambert ε and Δε |
| `fit_gaussians(...)` | fit Gaussian bands to one or more y columns simultaneously |
| `check_plot(id)` | overlay a fit on the data |

### MCD — adds field and temperature

| Method | Does |
|---|---|
| `subtract(ref_id=-1)` | subtract the appropriate 0 T scan (`-1` = the preceding one, `+1` = the following one), matched by temperature |
| `subtract_component(other, scalar=)` | subtract a scaled component spectrum at matching field/temperature |
| `half_subd_fields()` | `0.5*(+T − −T)` as an alternative to 0 T subtraction |
| `check_mirroring()` | `+T + −T`, which should go to zero |
| `loc_field(field)` | slice to `'+'`, `'-'`, or specific field values |
| `add_satmag_x()` / `add_satmag_y(x_val)` | βH/2kT and normalized intensity for VTVH plots |
| `plt_satmag(x_val)` | matplotlib saturation-magnetization plot, isotherms or isofields |
| `fit_doublet_model_vtvh(...)` | fit VTVH data to the doublet model |
| `make_satmag_fit_csv()` / `make_simpoints_csv()` | export for the external VTVH fitting/simulation codes |
| `quick_plot(..., field=, temp=)` | as the base, plus field/temperature filtering |

### VTVH_MCD — interval-scan mode

| Method | Does |
|---|---|
| `load(...)` | parse multi-channel interval-scan exports, averaging replicates into `_avg`/`_std` columns |
| `average_replicates()` | combine scans at the same field and temperature into a new object |

### DFT — cclib output

| Method | Does |
|---|---|
| `get_param(id, param, file_ind=)` | any cclib attribute; `file_ind` selects the job stage (0=opt, 1=sp, 2=freq) |
| `get_geom(id)` | atoms, final geometry, and Mulliken charges/spins as one frame |
| `get_dist(id, a1, a2[, a3])` | bond distance, or the a1–a2–a3 angle when `a3` is given |
| `get_dihedral(id, a1, a2, a3, a4)` | dihedral angle in degrees |
| `get_vibs(id)` | frequencies and IR intensities |
| `get_thermo(id, reactant_id=)` | E, E+ZPVE, H, S, G — absolute, or as a delta vs. a reactant |
| `get_scf(id)` | SCF energy in the requested units |

## A typical MCD session

```python
from sollabdata import MCD

data = MCD(
    path_to_raw_data="./folder_with_data/",
    info_csv="vtvh_log.csv",
)

data.add_wavenums()
with_zeros = data.copy()        # keep the 0 T scans before they are consumed
data.subtract(ref_id=-1)        # subtract the preceding 0 T scan at each temperature

fig = data.quick_plot(x=data.labels["NM"], y=data.labels["CD"])
fig.show()

```

## Project layout

```
src/sollabdata/
    labdata.py      LabData  — the base class and the info_df model
    abscd.py        AbsCD    — J-1700 Abs/CD
    mcd.py          MCD, VTVH_MCD
    dft.py          DFT
tests/
    test_abscd.py
    test_dft.py
    data/           committed fixtures: real J-1700 exports and a cclib json
```

## Testing

```bash
pytest                                   # whole suite
pytest tests/test_abscd.py               # one file
pytest tests/test_abscd.py::test_fit_gaussians    # one test
pytest -s                                # don't swallow print output
pytest --collect-only -q                 # list tests without running them
```

Fixtures live in `tests/data/`. The Abs/CD fixtures are real J-1700 exports of synthetic
spectra with known parameters — a +1 mdeg band at 600 nm (100 nm FWHM) and a −1 mdeg
band at 800 nm (200 nm FWHM) — so fits can be checked against ground truth. When adding
a fixture, keep it small and record the expected values in the fixture docstring. Use
pytest's `tmp_path` instead for deliberately malformed inputs.

## Known limitations

Things that will bite you, roughly in order of likelihood:

- **`subtract` does not check that x axes align.** It subtracts positionally. Spectra
  recorded over different ranges or step sizes will produce silent nonsense.
- **All files in one folder must share the same channel layout.** Channel labels are
  accumulated across every file as the folder is parsed, so a folder mixing exports with
  different `YUNITS` sets will mislabel columns.
- **Some methods assume a 0..n-1 index.** `drop()` re-indexes by default; if you use
  `reset_index=False` or slice `info_df` yourself, call
  `info_df.reset_index(drop=True, inplace=True)` before continuing.
- **`write()` reuses the `File` names from `info_df`,** so writing into the folder you
  read from overwrites your raw data. Always pass a new path.
- **`DFT.get_param` without `file_ind` returns the first non-list value it finds**,
  which is `NaN` for attributes only present on the frequency job. Pass `file_ind` (or
  `freq_idx`) explicitly for anything thermochemical.
- **`fit_gaussians` requires the initial guess to be inside any bounds you supply**,
  otherwise scipy raises `Initial guess is outside of provided bounds`.
- **`VTVH_MCD.average_replicates` needs `Date` and `Num_Temp_Checks` columns** in the
  info csv, which are not documented anywhere else.

## License

MIT — see [LICENSE](LICENSE).

<!--
Written by RG (Robert Gipson). Drafted by Claude (Opus 5) from the package source, the
test fixtures, and RG's MCD analysis notebook: install and dependency sections, a
quickstart verified against tests/data/abscd/, an explanation of the info_df data model,
per-class API tables, project layout, testing notes, and a known-limitations list.
-->
