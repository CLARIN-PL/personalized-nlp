from typing import Dict, Union, Tuple, List
import os
import glob
import json
import tqdm
import numpy as np


def read_data(path: str) -> Tuple[
        Dict[str, Dict[str, Union[int, np.ndarray]]],
        int
    ]:
    
    files = sorted(
        glob.glob(os.path.join(path, '*.jsonl')),
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    # open first to determine strategy
    train_dynamics = {}
    with open(files[0], 'r') as f:
        record = json.loads(next(f).strip())
        class_names = record['class_names']
        
    for class_name in class_names:
        train_dynamics[class_name] = {}
 
    for epoch, file in tqdm.tqdm(enumerate(files)):
        with open(file, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                # decode important messages
                guid = record['guid']
                class_names = record['class_names']
                predictions = record[f'logits_epoch_{epoch}']
                golds = record['gold']
                for class_name, gold, logits in zip(class_names, golds, np.split(np.array(predictions), len(class_names))):
                    if guid not in train_dynamics[class_name]:
                        assert epoch == 0
                        train_dynamics[class_name][guid] = {"gold": gold, "logits": []}
                    train_dynamics[class_name][guid]["logits"].append(logits)
                    
    return train_dynamics, len(files)
