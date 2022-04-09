import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1, Precision, Recall


class Classifier(pl.LightningModule):
    def __init__(self, model, class_dims, lr, nr_frozen_epochs, class_names=None):
        super().__init__()
        self.model = model
        self.lr = lr

        self.class_dims = class_dims
        self.class_names = class_names
        self.hparams.nr_frozen_epochs = nr_frozen_epochs

        self.metric_types = ('accuracy', 'precision',
                             'recall', 'f1', 'macro_f1')

        class_metrics = {}

        for split in ['train', 'valid', 'test']:
            for class_idx in range(len(class_dims)):
                class_name = class_names[class_idx] if class_names else str(
                    class_idx)

                num_classes = class_dims[class_idx]

                class_metrics[f'{split}_accuracy_{class_name}'] = Accuracy()
                class_metrics[f'{split}_precision_{class_name}'] = Precision(
                    num_classes=num_classes, average=None)
                class_metrics[f'{split}_recall_{class_name}'] = Recall(
                    num_classes=num_classes, average=None)
                class_metrics[f'{split}_f1_{class_name}'] = F1(
                    average='none', num_classes=num_classes)
                class_metrics[f'{split}_macro_f1_{class_name}'] = F1(
                    average='macro', num_classes=num_classes)

        self.metrics = nn.ModuleDict(class_metrics)

    def forward(self, x):
        x = self.model(x)
        return x

    def step(self, output, y):
        loss = 0
        class_dims = self.class_dims
        for cls_idx in range(len(class_dims)):
            start_idx = sum(class_dims[:cls_idx])
            end_idx = start_idx + class_dims[cls_idx]

            loss = loss + \
                   nn.CrossEntropyLoss()(
                       output[:, start_idx:end_idx], y[:, cls_idx].long())

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch

        output = self.forward(x)
        loss = self.step(output=output, y=y)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        preds = torch.argmax(output, dim=1)

        return {'loss': loss, 'preds': preds}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        output = self.forward(x)
        loss = self.step(output=output, y=y)

        self.log('valid_loss', loss, prog_bar=True)
        self.log_all_metrics(output=output, y=y, split='valid')

        return loss

    def validation_epoch_end(self, outputs):
        self.log_class_metrics_at_epoch_end('valid')

    def test_step(self, batch, batch_idx):
        x, y = batch

        output = self.forward(x)
        loss = self.step(output=output, y=y)

        self.log('test_loss', loss, prog_bar=True)
        self.log_all_metrics(output=output, y=y, split='test',
                             on_step=False, on_epoch=True)

        return {"loss": loss, 'output': output, 'y': y}

    def test_epoch_end(self, outputs) -> None:
        self.log_class_metrics_at_epoch_end('test')

    def freeze(self) -> None:
        for name, param in self.named_parameters():
            if 'fc' not in name: 
                param.requires_grad = False
        self.model.frozen = True

    def unfreeze(self) -> None:
        if self.model.frozen:
            for name, param in self.named_parameters():
                if 'fc' not in name: 
                    param.requires_grad = True
        self.model.frozen = False

    def on_epoch_start(self):
        if self.current_epoch < self.hparams.nr_frozen_epochs:
            self.freeze()

        if self.current_epoch >= self.hparams.nr_frozen_epochs:
            self.unfreeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_all_metrics(self, output, y, split, on_step=None, on_epoch=None):
        class_dims = self.class_dims
        class_names = self.class_names

        output = torch.softmax(output, dim=1)

        for cls_idx in range(len(class_dims)):
            start_idx = sum(class_dims[:cls_idx])
            end_idx = start_idx + class_dims[cls_idx]

            class_name = class_names[cls_idx] if class_names else str(
                cls_idx)

            log_dict = {}
            for metric_type in self.metric_types:
                metric_key = f'{split}_{metric_type}_{class_name}'
                metric_value = self.metrics[metric_key](
                    output[:, start_idx:end_idx].float(), y[:, cls_idx].int())

                if not metric_value.size():
                    # Log only metrics with single value (e.g. accuracy or metrics averaged over classes)
                    log_dict[metric_key] = self.metrics[metric_key]

            self.log_dict(log_dict, on_step=on_step,
                          on_epoch=on_epoch, prog_bar=True)

    def log_class_metrics_at_epoch_end(self, split):
        class_dims = self.class_dims
        class_names = self.class_names

        for cls_idx in range(len(class_dims)):
            class_name = class_names[cls_idx] if class_names else str(cls_idx)

            log_dict = {}
            for metric_type in self.metric_types:
                metric_key = f'{split}_{metric_type}_{class_name}'
                metric = self.metrics[metric_key]

                if metric.average in [None, 'none']:
                    metric_value = self.metrics[metric_key].compute()
                    for idx in range(metric_value.size(dim=0)):
                        log_dict[f'{metric_key}_{idx}'] = metric_value[idx]

                    self.metrics[metric_key].reset()

            self.log_dict(log_dict)
