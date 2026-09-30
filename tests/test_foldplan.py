import contextlib
from datetime import date, timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest

from foldplan import Sample, make_folds, read_samples
from foldplan.__main__ import main


def samples(count=10, delay=0):
    start = date(2026, 1, 1)
    return [Sample(start + timedelta(days=i), start + timedelta(days=i + delay))
            for i in range(count)]


class FoldTests(unittest.TestCase):
    def test_expanding_history_and_nonoverlapping_test_blocks(self):
        folds = make_folds(samples(), initial_train_size=4, test_size=2)
        self.assertEqual([f['train_indices'] for f in folds],
                         [list(range(4)), list(range(6)), list(range(8))])
        self.assertEqual([f['test_indices'] for f in folds], [[4, 5], [6, 7], [8, 9]])

    def test_label_available_on_test_start_is_excluded(self):
        fold = make_folds(samples(delay=2), initial_train_size=4, test_size=2)[0]
        self.assertEqual(fold['train_indices'], [0, 1])
        self.assertEqual(fold['purged_indices'], [2, 3])

    def test_gap_is_separate_from_unavailable_labels(self):
        fold = make_folds(samples(delay=3), initial_train_size=4, test_size=2, gap=1)[0]
        self.assertEqual(fold['train_indices'], [0, 1])
        self.assertEqual(fold['purged_indices'], [2, 3])
        self.assertEqual(fold['gap_indices'], [4])
        self.assertEqual(fold['test_indices'], [5, 6])

    def test_nonmonotonic_availability_dates(self):
        data = samples(7)
        data[0] = Sample(data[0].date, date(2026, 2, 1))
        fold = make_folds(data, initial_train_size=4, test_size=2)[0]
        self.assertEqual(fold['train_indices'], [1, 2, 3])
        self.assertEqual(fold['purged_indices'], [0])

    def test_partial_tail_is_explicit(self):
        args = dict(initial_train_size=4, test_size=2)
        self.assertEqual(len(make_folds(samples(9), **args)), 2)
        self.assertEqual(make_folds(samples(9), keep_partial=True, **args)[-1]['test_indices'], [8])

    def test_rolling_window_keeps_most_recent_eligible_rows(self):
        data = samples(10)
        data[4] = Sample(data[4].date, date(2026, 2, 1))
        fold = make_folds(data, initial_train_size=6, test_size=2, max_train_size=3)[0]
        self.assertEqual(fold['train_indices'], [2, 3, 5])
        self.assertEqual(fold['window_indices'], [0, 1])
        self.assertEqual(fold['purged_indices'], [4])

    def test_insufficient_history_or_no_known_labels_fails(self):
        for data, initial in [(samples(3), 4), (samples(5), 4), (samples(10, 100), 4)]:
            with self.subTest(initial=initial, length=len(data)):
                with self.assertRaises(ValueError):
                    make_folds(data, initial_train_size=initial, test_size=2)

    def test_invalid_parameters(self):
        for field, value in [('initial_train_size', 0), ('test_size', -1),
                             ('gap', -1), ('test_size', 1.5), ('gap', True),
                             ('keep_partial', 'yes'), ('max_train_size', 0),
                             ('max_train_size', True)]:
            args = dict(initial_train_size=4, test_size=2)
            args[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                make_folds(samples(), **args)

    def test_rejects_duplicate_unsorted_or_impossible_dates(self):
        data = samples(5)
        invalid = [[], list(reversed(data)), [data[0], data[0]],
                   [Sample(date(2026, 1, 2), date(2026, 1, 1))]]
        for records in invalid:
            with self.assertRaises(ValueError):
                make_folds(records, initial_train_size=1, test_size=1)

    def test_partition_and_availability_invariants(self):
        for delay in range(4):
            for gap in range(3):
                data = samples(18, delay)
                folds = make_folds(data, initial_train_size=5, test_size=4, gap=gap, keep_partial=True)
                tested = []
                for fold in folds:
                    train, purge, omitted, window, test = [fold[k] for k in
                        ['train_indices', 'purged_indices', 'gap_indices', 'window_indices', 'test_indices']]
                    combined = train + purge + omitted + window + test
                    self.assertEqual(sorted(combined), list(range(test[-1] + 1)))
                    self.assertEqual(len(combined), len(set(combined)))
                    self.assertTrue(all(data[i].available_on < data[test[0]].date for i in train))
                    tested.extend(test)
                self.assertEqual(len(tested), len(set(tested)))


class CsvAndCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'samples.csv'

    def write(self, content):
        self.path.write_text(content, encoding='utf8')

    def test_bom_extra_columns_and_input_preserved(self):
        content = '\ufeffdate,available_on,note\r\n2026-01-01,2026-01-02,"a,b"\r\n'
        self.write(content)
        before = self.path.read_bytes()
        self.assertEqual(read_samples(self.path), [Sample(date(2026, 1, 1), date(2026, 1, 2))])
        self.assertEqual(before, self.path.read_bytes())

    def test_bad_csv_is_rejected(self):
        for content in ['', 'date,available_on\n', 'date,date\n',
                        'date,available_on\n2026-01-01\n',
                        'date,available_on\n2026-02-30,2026-03-01\n',
                        'date,available_on\n20260101,2026-01-02\n',
                        'date,available_on\n"unterminated']:
            self.write(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_samples(self.path)

    def test_cli_json_and_indices(self):
        self.write('date,available_on\n' + ''.join(f'{s.date},{s.available_on}\n' for s in samples(6)))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main([str(self.path), '--initial-train-size', '3', '--test-size', '2'])
        self.assertEqual(result, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report['index_base'], 0)
        self.assertEqual(report['folds'][0]['test_indices'], [3, 4])

    def test_cli_failure_has_no_partial_json(self):
        for contents in [None, b'\xff', b'date,available_on\n']:
            if contents is not None:
                self.path.write_bytes(contents)
            output, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                result = main([str(self.path), '--initial-train-size', '3', '--test-size', '2'])
            self.assertEqual(result, 2)
            self.assertEqual(output.getvalue(), '')
            self.assertIn('Could not plan folds', errors.getvalue())
