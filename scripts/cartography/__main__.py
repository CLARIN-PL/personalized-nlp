import os
import argparse

from scripts.cartography.read_data import read_data
from scripts.cartography.plot_cartography import plot_data_map, compute_metrics, compute_avg_metrics

from settings import CARTOGRAPHY_TRAIN_DYNAMICS_DIR_NAME, CARTOGRAPHY_METRICS_DIR_NAME, CARTOGRAPHY_PLOTS_DIR_NAME


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
    
    train_dynamics_path = os.path.join(args.path, CARTOGRAPHY_TRAIN_DYNAMICS_DIR_NAME)
    plots_path = os.path.join(args.path, CARTOGRAPHY_PLOTS_DIR_NAME)
    metrics_path = os.path.join(args.path, CARTOGRAPHY_METRICS_DIR_NAME)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path) 
    
    train_dynamics, num_epochs = read_data(train_dynamics_path)

    metrics = []
    for class_name in train_dynamics.keys():
        train_dynamics_cls = train_dynamics[class_name]
        # print(train_dynamics_cls.keys())
        cls_metrics = compute_metrics(train_dynamics_cls, num_epochs=num_epochs)
        metrics.append(cls_metrics)
        plot_data_map(cls_metrics, plots_path, title=f'{class_name}')
    if len(metrics) > 1:
        avg_metrics = compute_avg_metrics(metrics)
        plot_data_map(avg_metrics, plots_path, title=f'average')
        
        

if __name__ == '__main__':
    main()
