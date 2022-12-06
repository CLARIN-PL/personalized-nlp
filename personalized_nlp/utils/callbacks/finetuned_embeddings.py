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
        model_cls = type(pl_module)
        pl_model_best = model_cls.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

        model = pl_model_best.model._model.to("cuda")
        tokenizer = pl_model_best.model._tokenizer
        texts = self.datamodule.data["text"].tolist()

        embeddings = _get_embeddings(
            texts, tokenizer, model, use_cuda=True, max_seq_len=128
        )
        embeddings = embeddings.cpu().numpy()

        text_idx_to_emb = {}
        for i in range(embeddings.shape[0]):
            text_idx_to_emb[i] = embeddings[i]

        embeddings_path = self.save_path
        if embeddings_path:
            pickle.dump(text_idx_to_emb, open(embeddings_path, "wb"))
