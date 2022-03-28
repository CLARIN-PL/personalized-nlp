import torch
import torch.nn as nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
import pytorch_lightning as pl
import shutil


class Regressor(pl.LightningModule):

    def __init__(self, model, lr, class_names, epochs=6, is_frozen=False):
        super().__init__()
        self.model = model
        self.lr = lr

        self.class_names = class_names
        self.is_frozen = is_frozen

        self.metric_types = ['r2']

        class_metrics = {}

        for split in ['train', 'valid', 'test']:
            for class_idx in range(len(class_names)):
                class_name = class_names[class_idx]

                class_metrics[f'{split}_mae_{class_name}'] = MeanAbsoluteError(
                )
                class_metrics[f'{split}_mse_{class_name}'] = MeanSquaredError()
                class_metrics[f'{split}_r2_{class_name}'] = R2Score()

        self.metrics = nn.ModuleDict(class_metrics)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()

        output = self.forward(x)

        loss = nn.MSELoss()(output.float(), y.float())

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)

        return loss

    def training_epoch_end(self, outs):
        pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()

        output = self.forward(x)

        loss = nn.MSELoss()(output, y)

        self.log('valid_loss', loss, prog_bar=True)
        self.log_all_metrics(output=output, y=y, split='valid')

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()

        output = self.forward(x)

        loss = nn.MSELoss()(output, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log_all_metrics(output=output,
                             y=y,
                             split='test',
                             on_step=False,
                             on_epoch=True)

        return {"loss": loss, 'output': output, 'y': y}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if (self.is_frozen):
            model = self.model
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                    self.hparams.weight_decay,
                },
                {
                    "params": [
                        p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                    0.0,
                },
            ]
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.lr)

        return optimizer

    def log_all_metrics(self, output, y, split, on_step=None, on_epoch=None):
        class_names = self.class_names

        if len(output) >= 2:
            log_dict = {}
            for metric_type in self.metric_types:
                metric_values = []
                for class_idx, class_name in enumerate(class_names):
                    metric_key = f'{split}_{metric_type}_{class_name}'
                    metric_value = self.metrics[metric_key](output[:,
                                                                   class_idx],
                                                            y[:, class_idx])

                    metric_values.append(metric_value)
                    log_dict[metric_key] = metric_value

                mean_metric_key = f'{split}_{metric_type}_mean'
                log_dict[mean_metric_key] = torch.mean(
                    torch.tensor(metric_values))

                self.log_dict(log_dict, on_step=on_step, on_epoch=on_epoch)
