import pickle
import time

from personalized_nlp.utils.embeddings import _get_embeddings
from pytorch_lightning.callbacks import Callback


class SaveEmbeddingCallback(Callback):
    def __init__(self, datamodule, save_path):
        self.datamodule = datamodule
        self.save_path = save_path

    def on_fit_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.epoch_stats = []

    def on_test_end(self, trainer, pl_module) -> None:
        print('on test end predictions')
        model = pl_module.model._model 
        tokenizer = pl_module.model._tokenizer
        texts = self.datamodule.data['text'].tolist()

        embeddings = _get_embeddings(texts, tokenizer, model, use_cuda=True)
        embeddings = embeddings.cpu().numpy()

        text_idx_to_emb = {}
        for i in range(embeddings.shape[0]):
            text_idx_to_emb[i] = embeddings[i]
        
        embeddings_path = self.save_path
        if embeddings_path:
            pickle.dump(text_idx_to_emb, open(embeddings_path, 'wb'))