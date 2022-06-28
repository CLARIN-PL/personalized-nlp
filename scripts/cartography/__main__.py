from typing import Dict, Union, Iterable
import os
import argparse

import json
import pandas as pd

from scripts.cartography.read_data import read_data
from scripts.cartography.plot_cartography import plot_data_map, compute_metrics, compute_avg_metrics, write_filtered_data


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments

    Returns:
        argparse.Namespace: arguments passed via command-line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--meta_path',
        '-p',
        dest='meta_path',
        type=str,
        required=True,
        help='Path to meta json file.'
    )
    parser.add_argument(
        '--start_fold',
        '-sf',
        dest='start_fold',
        type=int,
        default=0,
        required=False,
        help='Start fold, must be lesser than fold_nums'
    )
    parser.add_argument(
        '--sorted_by',
        '-sb',
        dest='sorted_by',
        type=str,
        choices=['variability'],
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
        help='Size of ambigous data to keep.'
    )
    return parser.parse_args()


def read_meta(path: str) -> Dict[str, Union[float, str]]:
    with open(path, 'r') as f:
        meta_dir = json.loads(f.read())
    return meta_dir


def create_all_paths(paths: Iterable[str]) -> None:
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def main() -> None:
    args = parse_args()
    meta_dir = read_meta(args.meta_path)
    
    for fold in range(args.start_fold, meta_dir["fold_nums"]):
    
        # paths
        train_dir = os.path.join(meta_dir["train_dir"], f'fold_{fold}')
        train_dynamics, num_epochs = read_data(train_dir)

        metrics = []
        for class_name in train_dynamics.keys():
            
            plots_dir = os.path.join(meta_dir["plots_dir"], f'class_{class_name}')
            metrics_dir = os.path.join(meta_dir["metrics_dir"], f'class_{class_name}')
            filter_dir = os.path.join(meta_dir["filter_dir"], f'class_{class_name}')
        
    
            create_all_paths(
                paths=(
                    plots_dir,
                    metrics_dir,
                    filter_dir
                )
            )
        
            
            train_dynamics_cls = train_dynamics[class_name]
            
            cls_metrics, df_train = compute_metrics(train_dynamics_cls, num_epochs=num_epochs)
            
            metrics.append(cls_metrics)
            plot_data_map(
                cls_metrics, 
                plots_dir, 
                save_name=f'{class_name}_fold_{fold}'
            )
            
            cls_metrics.to_csv(
                os.path.join(metrics_dir, f'{class_name}_fold_{fold}_metrics.csv'), 
                index=False
            )
            
            
            write_filtered_data(
                cls_metrics, 
                save_path=os.path.join(filter_dir, f'{class_name}_{args.sorted_by}_fold_{fold}_filtered.csv'), 
                sorted_by=args.sorted_by, 
                take_size=args.take_size
            )
            
        # if list of metrics contains multiple elements, create data map using average metrics
        if len(metrics) > 1:
            avg_metrics, df_train = compute_avg_metrics(metrics)
            plot_data_map(avg_metrics, meta_dir["plots_dir"], title=f'fold_{fold}_average')
            avg_metrics.to_csv(os.path.join(meta_dir["metrics_dir"], f'fold_{fold}_average_metrics.csv'), index=False)
            write_filtered_data(
                avg_metrics, 
                save_path=os.path.join(meta_dir["filter_dir"], f'fold_{fold}_average_{args.sorted_by}_filtered.csv'), 
                sorted_by=args.sorted_by, 
                take_size=args.take_size
            )
            
        

if __name__ == '__main__':
    main()
