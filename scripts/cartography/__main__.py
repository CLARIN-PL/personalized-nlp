import argparse

from scripts.cartography.read_data import read_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        '-p',
        dest='path',
        type=str,
        required=True,
        help='Path to directory with training history.'
    )
    parser.add_argument(
        '--multilabel',
        '-mlb',
        dest='multilabel',
        help='Whether to create multilabel and avg map'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    read_data(args.path)


if __name__ == '__main__':
    main()
