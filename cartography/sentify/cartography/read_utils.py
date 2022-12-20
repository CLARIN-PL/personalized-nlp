import json
import os
from pathlib import Path

from tqdm.auto import tqdm


def read_training_dynamics(
    train_dynamics_dir: Path,
    sample_id: str = 'guid',
    gold_label: str = 'gold',
    epoch_burn_out: int = None,
) -> dict[int, dict]:
    """
    Given path to logged training dynamics, merge stats across epochs.
    Returns:
    - dict between ID of a train instances and its gold label, and the list of logits across epochs.
    """
    train_dynamics = dict()

    num_epochs = len(list(train_dynamics_dir.iterdir()))
    if epoch_burn_out and epoch_burn_out < num_epochs:
        num_epochs = epoch_burn_out

    for epoch_num in tqdm(
        range(num_epochs),
        total=num_epochs,
        desc="Reading training dynamics log files",
    ):
        epoch_file = train_dynamics_dir.joinpath(f'dynamics_epoch_{epoch_num}.jsonl')
        assert os.path.exists(epoch_file), f"Missing train dynamics file for epoch: {epoch_num}"

        with epoch_file.open(mode='r') as infile:
            lines = infile.readlines()

            for line in lines:
                record = json.loads(line.rstrip('\n').strip())
                guid = record[sample_id]

                if guid not in train_dynamics:
                    assert epoch_num == 0, f"Sample id {guid} not appeared in epoch 0!"
                    train_dynamics[guid] = {
                        gold_label: record[gold_label],
                        'logits': [],
                        'epochs': num_epochs,
                    }
                train_dynamics[guid]['logits'].append(record[f'logits_epoch_{epoch_num}'])

    return train_dynamics
