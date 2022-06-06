import os
import glob
import json


def read_data(path: str):
    files = sorted(
        glob.glob(os.path.join(path, '*.jsonl')),
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    # open first to determine strategy
    train_dynamics = {}
    with open(files[0], 'r') as f:
        record = json.loads(next(f).strip())
        class_names = record['class_names']
        num_classes = len(record['logits_epoch_0'])
    for class_name in class_names:
        train_dynamics[class_name] = {}
 
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                class_names = record['class_names']
                if len(train_dynamics.keys()) == 0:
                    for class_name in clas
