import os

import pandas as pd
import torch
from math import ceil
from sklearn.metrics import classification_report
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm

from sentify import EXPERIMENT_DIR, DATASETS_PATH, RESULTS_PATH
from sentify.models.models import TransformerSentimentModel, RetrieverModel
from sentify.utils.config import load_config
from sentify.utils.experiments import (
    create_datamodule,
    create_user_identifier_datamodule,
    create_retriever_datamodule,
)

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

FILENAMES = {
    'sentiment140_baseline': 'epoch=3-step=4352.ckpt',  # good (amd-worker)
    'sentiment140_retriever': 'epoch=1-step=2176.ckpt',  # good (amd-worker)
    'sentiment140_user_identifier': 'epoch=23-step=26112.ckpt',  # good (amd-worker)
    'MHS_baseline': 'epoch=1-step=1823.ckpt',  # good
    'MHS_retriever': 'epoch=11-step=10943.ckpt',  # good
    'MHS_user_identifier': 'epoch=12-step=11856.ckpt',  # good (amd-worker)
    'imdb_baseline': 'epoch=27-step=7000.ckpt',  # good (carrot)
    'imdb_retriever': 'epoch=33-step=8499.ckpt',  # good
    'imdb_user_identifier': 'epoch=15-step=4000.ckpt',  # good (amd-worker)
}

METHODS = {
    'baseline': (TransformerSentimentModel, create_datamodule),
    'user_identifier': (
        TransformerSentimentModel,
        create_user_identifier_datamodule,
    ),
    'retriever': (RetrieverModel, create_retriever_datamodule),
}

method = 'retriever'
dataset = 'MHS'
filename_key = f'{dataset}_{method}'
checkpoint_path = EXPERIMENT_DIR.joinpath(filename_key, 'checkpoints', FILENAMES[filename_key])

config = load_config()
config['datamodule']['dataset'] = dataset
config['method'] = method

model_cls, datamodule_func = METHODS[method]
datamodule = datamodule_func(config)

print(f'Loading from checkpoint: {checkpoint_path}')
model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint_path)
model.eval()
print('Loaded checkpoint')

df_test = pd.read_csv(DATASETS_PATH.joinpath(dataset, 'test.tsv'), sep='\t')
true_labels = df_test['label'].values.tolist()

metric_acc = Accuracy(average='micro')
metric_f1 = F1Score(average='macro', num_classes=datamodule.num_classes)

predictions = []
for idx, batch in tqdm(
    enumerate(datamodule.test_dataloader()),
    desc="Make predict on test split...",
    total=ceil(len(true_labels) / config['datamodule']['batch_size']),
):
    y_hat = model.test_step(batch, batch_idx=idx)
    predicted_logits = y_hat['logits']
    predicted_labels = [torch.argmax(logit).item() for logit in predicted_logits]

    metric_acc.update(preds=predicted_logits, target=y_hat['labels'])
    metric_f1.update(preds=predicted_logits, target=y_hat['labels'])

    predictions.extend(predicted_labels)

accuracy = metric_acc.compute().item()
f1_macro = metric_f1.compute()
print(f'Accuracy: {accuracy}')
print(f'F1 macro: {f1_macro}')

report = classification_report(
    y_true=true_labels,
    y_pred=predictions,
)
print(report)

metric_path = RESULTS_PATH.joinpath(f'{filename_key}_metrics.txt')
with metric_path.open(mode='w') as f:
    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'F1 macro: {f1_macro}\n')

df_test['predicted'] = predictions
df_test.to_csv(RESULTS_PATH.joinpath(f'{filename_key}.csv'), index=False)
