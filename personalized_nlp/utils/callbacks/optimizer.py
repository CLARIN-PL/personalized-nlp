import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from personalized_nlp.learning.classifier import Classifier


class SetLearningRates(Callback):
    def __init__(
        self,
        lr: float = 1e-5,
        new_embeddings_lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.new_embeddings_lr = new_embeddings_lr

    def on_train_start(self, trainer: Trainer, pl_module: Classifier):
        model = pl_module.model

        param_optimizer = list(model.named_parameters())
        embedding_names = 'emotion_embeddings', 'sentiment_embeddings'
        params = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in embedding_names)],
                'lr': self.lr,
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in embedding_names)],
                'lr': self.new_embeddings_lr,
            },
        ]
        print([len(p['params']) for p in params])

        optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
        )
        trainer.optimizers = [optimizer]


class SetWeightDecay(Callback):
    def __init__(
        self,
        lr: float = 1e-5,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay

    def on_train_start(self, trainer: Trainer, pl_module: Classifier):
        model = pl_module.model
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        trainer.optimizers = [optimizer]
