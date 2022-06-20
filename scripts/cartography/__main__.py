import os
import argparse

import pandas as pd

from scripts.cartography.read_data import read_data
from scripts.cartography.plot_cartography import plot_data_map, compute_metrics, compute_avg_metrics, write_filtered_data

from settings import CARTOGRAPHY_FILTER_DIR_NAME, CARTOGRAPHY_TRAIN_DYNAMICS_DIR_NAME, CARTOGRAPHY_METRICS_DIR_NAME, CARTOGRAPHY_PLOTS_DIR_NAME, CARTOGRAPHY_FILTER_DIR_NAME


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments

    Returns:
        argparse.Namespace: arguments passed via command-line
    """
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
        '--sorted_by',
        '-sb',
        dest='sorted_by',
        type=str,
        required=False,
        default='variability',
        help='Group to save first'
    )
    parser.add_argument(
        '--take_size',
        '-ts',
        dest='take_size',
        type=float,
        default=0.3,
        required=False,
        help='Size'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    train_dynamics_path = os.path.join(args.path, CARTOGRAPHY_TRAIN_DYNAMICS_DIR_NAME)
    plots_path = os.path.join(args.path, CARTOGRAPHY_PLOTS_DIR_NAME)
    metrics_path = os.path.join(args.path, CARTOGRAPHY_METRICS_DIR_NAME)
    filters_path = os.path.join(args.path, CARTOGRAPHY_FILTER_DIR_NAME)
    
    # TODO: move to separate function
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path) 
    if not os.path.exists(filters_path):
        os.makedirs(filters_path)
    
    train_dynamics, num_epochs = read_data(train_dynamics_path)

    metrics = []
    for class_name in train_dynamics.keys():
        train_dynamics_cls = train_dynamics[class_name]
        cls_metrics, df_train = compute_metrics(train_dynamics_cls, num_epochs=num_epochs)
        print(df_train)
        metrics.append(cls_metrics)
        plot_data_map(cls_metrics, plots_path, title=f'{class_name}')
        cls_metrics.to_csv(os.path.join(metrics_path, f'{class_name}_metrics.csv'), index=False)
        write_filtered_data(
            cls_metrics, 
            save_path=os.path.join(filters_path, f'{class_name}_{args.sorted_by}_filtered.csv'), 
            sorted_by=args.sorted_by, 
            take_size=args.take_size
        )
        
    # if list of metrics contains multiple elements, create data map using average metrics
    if len(metrics) > 1:
        avg_metrics, df_train = compute_avg_metrics(metrics)
        plot_data_map(avg_metrics, plots_path, title=f'average')
        avg_metrics.to_csv(os.path.join(metrics_path, f'average_metrics.csv'), index=False)
        write_filtered_data(
            avg_metrics, 
            save_path=os.path.join(filters_path, f'average_{args.sorted_by}_filtered.csv'), 
            sorted_by=args.sorted_by, 
            take_size=args.take_size
        )
        
        

if __name__ == '__main__':
    main()
