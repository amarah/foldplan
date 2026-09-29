import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    date: date
    available_on: date


def parse_date(value: str, field: str, row: int) -> date:
    try:
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            raise ValueError
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(f'Record {row}: {field} must be a valid YYYY-MM-DD date.') from None


def read_samples(path: str | Path) -> list[Sample]:
    """Read dates without sorting or altering the source CSV."""
    samples = []
    with open(path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader, None)
            if header is None:
                raise ValueError('CSV is empty.')
            header = [name.strip() for name in header]
            if not all(header) or len(set(header)) != len(header):
                raise ValueError('Column names must be nonempty and unique.')
            if not {'date', 'available_on'}.issubset(header):
                raise ValueError('CSV requires date and available_on columns.')
            for row, cells in enumerate(reader, 2):
                if len(cells) != len(header):
                    raise ValueError(f'Record {row}: expected {len(header)} fields; got {len(cells)}.')
                values = dict(zip(header, (cell.strip() for cell in cells)))
                samples.append(Sample(parse_date(values['date'], 'date', row),
                                      parse_date(values['available_on'], 'available_on', row)))
        except csv.Error as exc:
            raise ValueError(f'Malformed CSV near line {reader.line_num}: {exc}') from exc
    validate_samples(samples)
    return samples


def validate_samples(samples):
    if not samples:
        raise ValueError('At least one sample is required.')
    previous = None
    for sample in samples:
        if not isinstance(sample, Sample) or type(sample.date) is not date or type(sample.available_on) is not date:
            raise ValueError('Samples must contain date objects, without intraday timestamps.')
        if previous is not None and sample.date <= previous:
            raise ValueError('Sample dates must be unique and strictly increasing.')
        if sample.available_on < sample.date:
            raise ValueError('Label availability cannot precede its sample date.')
        previous = sample.date


def make_folds(samples, *, initial_train_size: int, test_size: int,
               gap: int = 0, keep_partial: bool = False) -> list[dict]:
    """Return expanding-window splits with zero-based indices into samples.

    Labels must be available strictly before the first test date. The optional
    gap excludes additional rows immediately before each test window.
    """
    samples = list(samples)
    validate_samples(samples)
    for name, value, minimum in [('initial_train_size', initial_train_size, 1),
                                 ('test_size', test_size, 1), ('gap', gap, 0)]:
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer of at least {minimum}.')
    if type(keep_partial) is not bool:
        raise ValueError('keep_partial must be a boolean.')
    folds = []
    for start in range(initial_train_size + gap, len(samples), test_size):
        stop = min(start + test_size, len(samples))
        if stop - start < test_size and not keep_partial:
            break
        cutoff = samples[start].date
        candidates = range(start - gap)
        train = [i for i in candidates if samples[i].available_on < cutoff]
        purged = [i for i in candidates if samples[i].available_on >= cutoff]
        if not train:
            raise ValueError(f'No training labels are available before {cutoff.isoformat()}.')
        folds.append({
            'fold': len(folds) + 1,
            'test_start': cutoff.isoformat(),
            'test_end': samples[stop - 1].date.isoformat(),
            'train_indices': train,
            'test_indices': list(range(start, stop)),
            'gap_indices': list(range(start - gap, start)),
            'purged_indices': purged,
        })
    if not folds:
        raise ValueError('Not enough samples for a test fold; reduce sizes or use keep_partial.')
    return folds
