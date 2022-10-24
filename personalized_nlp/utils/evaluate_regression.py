import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
from personalized_nlp.utils.finetune import finetune_datamodule_embeddings
from personalized_nlp.learning.regressor import Regressor
import tqdm
import torch


CHECKPOINT_PATH = "/home/mgruza/repos/personalized-nlp/storage/checkpoints/feasible-frost-11/epoch=1-step=5580.ckpt"
TEXTS_PATH = "/home/mgruza/repos/personalized-nlp/storage/other/sd_data.tsv"
PREDICTION_PATH = "/home/mgruza/repos/personalized-nlp/storage/other/sd_data_aggression_entropy_predictions.csv"

USE_CUDA = False


def batch_forward(classifier, texts, batch_size=10):
    def batch(iterable, n=batch_size):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx : min(ndx + n, l)]

    probabs = []
    for text_batch in tqdm.tqdm(batch(texts)):
        annotators = np.array([0] * text_batch.shape[0])
        batch_probabs = classifier.forward(
            {"raw_texts": text_batch, "annotator_ids": annotators}
        )
        probabs.extend(batch_probabs.unsqueeze(0))

    return torch.cat(probabs, dim=0)


if __name__ == "__main__":

    map_location = "cuda" if USE_CUDA else "cpu"
    regressor = Regressor.load_from_checkpoint(
        CHECKPOINT_PATH, map_location=map_location
    )
    regressor.model.use_cuda = USE_CUDA
    regressor = regressor.eval()

    texts = pd.read_csv(TEXTS_PATH, sep="\t")["text"].values

    with torch.no_grad():
        predictions = batch_forward(regressor, texts)

    result_df = pd.DataFrame({"text": texts})
    for idx, col in enumerate(regressor.class_names):
        result_df[col] = predictions[:, idx]

    result_df.to_csv(PREDICTION_PATH, index=False)
