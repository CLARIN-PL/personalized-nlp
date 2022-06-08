import argparse

from scripts.cartography.read_data import read_data
from scripts.cartography.plot_cartography import plot_data_map, compute_metrics


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
    train_dynamics, num_epochs = read_data(args.path)
    for class_name in train_dynamics.keys():
        train_dynamics_cls = train_dynamics[class_name]
        # print(train_dynamics_cls.keys())
        cls_metrics = compute_metrics(train_dynamics_cls, num_epochs=num_epochs)
        plot_data_map(cls_metrics, '.', title=f'{class_name}')
        
        

if __name__ == '__main__':
    main()
