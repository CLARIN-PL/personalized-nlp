from sched import scheduler
import torch
import torch.nn as nn
from personalized_nlp.learning.regressor import Regressor
from personalized_nlp.utils.callbacks.optimizer import SetLearningRates, SetWeightDecay
from personalized_nlp.utils.callbacks.transformer_lr_scheduler import TransformerLrScheduler
from transformers import get_linear_schedule_with_warmup

import pytorch_lightning as pl


class RegressorFinetune(Regressor):
    def __init__(self, model, lr, class_names):
        super().__init__(model, lr, class_names)

    def configure_optimizers(self):
      model = self.model
      param_optimizer = list(model.named_parameters())

      no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
      optimizer_grouped_parameters = [
          {
              "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
              "weight_decay": self.hparams.weight_decay,
          },
          {
              "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
              "weight_decay": 0.0,
          },
      ]
      optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
      return [optimizer]