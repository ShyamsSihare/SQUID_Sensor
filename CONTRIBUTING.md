# Contributing

Thanks for considering a contribution to this simulation/figure-generation
pipeline. A few ground rules keep the project's core promise intact: **every
number in this repository must be traceable to an equation and a script**,
never hand-typed.

## Ground rules

1. **No hand-typed figure or table values.** If a number appears in a
   figure, a table, or a caption, there must be a line of code in
   `src/squid_sim/` or `notebooks/modules/` that computes it from the
   dataset or directly from the governing equations.
2. **Disclose, don't silently fix.** If you find another unit inconsistency
   or implausible parameter combination in the manuscript's equations (see
   `docs/METHODOLOGY.md` for the three already found), document your
   calibration choice the same way the existing ones are documented —
   in the module docstring, in `docs/METHODOLOGY.md`, and in the README's
   "Known limitations" section — rather than quietly changing a number.
3. **Fixed seeds.** Any new stochastic simulation must take an explicit
   `seed` parameter and default to a value derived from `MASTER_SEED` in
   `run_all.py`, so results are bit-reproducible.
4. **Validate what you add.** If you add a new plotted or tabulated
   quantity, add a corresponding check to
   `notebooks/modules/module_02_validation_suite.py` (and
   `tests/test_validation_suite.py`) that independently re-derives it.

## Development workflow

```bash
git clone https://github.com/OWNER/REPO.git
cd REPO
mamba env create -f environment.yml && conda activate squid-sim

# after making changes to src/squid_sim/:
python src/squid_sim/run_all.py

# after making changes to notebooks/modules/*.py:
cd notebooks && python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace SQUID_Manuscript_Figures.ipynb

# before opening a PR:
pytest tests/ -v
flake8 src/ notebooks/modules/ --max-line-length=110
```

## Pull request checklist

- [ ] `pytest tests/` passes locally
- [ ] The notebook executes end-to-end with **zero** cell errors
      (`jupyter nbconvert --to notebook --execute`)
- [ ] Any new numeric claim has a corresponding validation-suite check
- [ ] Any new calibration constant or manuscript-equation deviation is
      documented in `docs/METHODOLOGY.md`
- [ ] `README.md`'s "Headline results" table is updated if your change
      alters any of those numbers

## Reporting issues

Please use the issue templates under `.github/ISSUE_TEMPLATE/` — there is a
dedicated template for reporting a suspected numerical/physics
inconsistency, since that is the failure mode this project cares most about
catching early.
