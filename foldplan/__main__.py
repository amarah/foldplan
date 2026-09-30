import argparse
import json
import sys

from .core import make_folds, read_samples


def main(argv=None):
    parser = argparse.ArgumentParser(description='Plan chronological train/test folds from label availability dates.')
    parser.add_argument('file', help='CSV with date and available_on columns')
    parser.add_argument('--initial-train-size', required=True, type=int)
    parser.add_argument('--test-size', required=True, type=int)
    parser.add_argument('--gap', default=0, type=int, help='Rows excluded immediately before each test block')
    parser.add_argument('--max-train-size', type=int,
                        help='Keep only this many of the most recent eligible training rows')
    parser.add_argument('--keep-partial', action='store_true', help='Include a final shorter test block')
    args = parser.parse_args(argv)
    try:
        samples = read_samples(args.file)
        folds = make_folds(samples, initial_train_size=args.initial_train_size,
                           test_size=args.test_size, gap=args.gap, keep_partial=args.keep_partial,
                           max_train_size=args.max_train_size)
    except (ValueError, OSError, UnicodeError) as exc:
        print(f'Could not plan folds: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({'schema_version': 1, 'sample_count': len(samples),
                      'index_base': 0, 'folds': folds}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
