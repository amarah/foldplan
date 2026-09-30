# foldplan

Generate chronological train/test splits from a CSV of sample dates and label
availability dates. Use it when a target, such as a future return, is not known
on the day its features were recorded.

The output records exactly which rows enter training and testing, which rows
are excluded by a gap, and which training labels are still unavailable. It
does not train a model or change your data.

## Run it

Python 3.10 or later. No runtime dependencies or network calls.
Unzip the package and run these commands from the folder containing this README.

```bash
python -m foldplan examples/labels.csv --initial-train-size 4 --test-size 2
python -m unittest discover -s tests -v
```

To install the command in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
foldplan examples/labels.csv --initial-train-size 4 --test-size 2 --gap 1
foldplan examples/labels.csv --initial-train-size 4 --test-size 2 --max-train-size 3
```

On Windows, activate with `.venv\Scripts\activate` instead. Installation may
download setuptools if it is missing. The project is not published to PyPI.

## Input

```csv
date,available_on
2026-01-05,2026-01-07
2026-01-06,2026-01-08
2026-01-07,2026-01-09
2026-01-08,2026-01-12
2026-01-09,2026-01-13
2026-01-12,2026-01-14
```

These are synthetic dates. `date` identifies a sample. `available_on` is the
first date its target is known. Supply actual availability dates, including
any publication delay. The tool does not calculate them or consult an exchange
calendar. Extra columns are ignored. Required column names are case-sensitive.
Dates use YYYY-MM-DD and sample dates must be unique and strictly increasing.
Availability cannot precede the sample date, but can extend beyond the dataset.

## Split rules

- Start the first test block after `initial_train_size + gap` rows.
- Advance by `test_size` rows for each new test block. Test blocks do not overlap.
- Expand candidate training history from the start of the dataset.
- If `--max-train-size` is set, keep only that many of the most recent eligible
  training rows. This creates a rolling window after unavailable labels are purged.
- Exclude the `gap` rows immediately before each test block.
- From remaining candidates, exclude labels whose availability date is on or
  after the first test date. Same-day availability is conservatively excluded.
- Drop a final shorter test block unless `--keep-partial` is supplied.

`initial_train_size` is the number of candidate training rows, not a minimum
number remaining after exclusions. A fold with no usable training rows causes
an error. Earlier test rows can enter training in later folds once their labels
are available. A gap counts observations, not calendar days.

With the six rows above, `--initial-train-size 4 --test-size 2` produces training
indices `[0, 1]`, purged indices `[2, 3]`, and test indices `[4, 5]`.

## Output and Python API

The CLI prints JSON with `schema_version`, `sample_count`, `index_base`, and
`folds`. Each fold includes its test date range and `train_indices`,
`test_indices`, `gap_indices`, and `purged_indices`. All indices are zero-based
data-row positions, excluding the CSV header. `window_indices` lists older,
otherwise eligible rows excluded by `--max-train-size`. No automatic sorting occurs.
Exit code 0 means success; 2 means invalid arguments, input, or file access.
Errors go to stderr and no partial JSON is emitted.

```python
from foldplan import make_folds, read_samples

samples = read_samples('examples/labels.csv')
folds = make_folds(samples, initial_train_size=4, test_size=2)
for fold in folds:
    print(fold['train_indices'], fold['test_indices'])
```

Use those positions with your own arrays or `frame.iloc[...]`. Keep the data in
the original row order. Fit preprocessing and the model using each fold's
training rows only, then evaluate on that fold's test rows.

## Limits

This is a single-series, date-only planner. It does not handle multiple symbols
sharing a date, intraday availability, rolling training windows, or overlapping
test windows. It stores the input and all fold indices in memory.

Availability filtering alone cannot guarantee a leakage-free backtest. Incorrect
availability dates, revised data, features computed using future observations,
and preprocessing fitted on the full dataset can still leak information. The
planner does not inspect features or evaluate strategy quality.
