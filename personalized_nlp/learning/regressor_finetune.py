import torch
import torch.nn as nn
import pytorch_lightning as pl
from personalized_nlp.learning.regressor import Regressor


class RegressorFinetune(Regressor):
    def __init__(self, model, lr, weight_decay, class_names):
        super().__init__(model, lr, class_names)
        self.weight_decay = weight_decay

    def configure_optimizers(self):
      model = self.model
      param_optimizer = list(model.named_parameters())

      no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
      optimizer_grouped_parameters = [
          {
              "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
              "weight_decay": self.weight_decay,
          },
          {
              "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
              "weight_decay": 0.0,
          },
      ]
      optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
      lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
      return [optimizer], [lr_scheduler]
