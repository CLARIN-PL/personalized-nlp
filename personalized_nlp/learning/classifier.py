import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall, F1


class Classifier(pl.LightningModule):
    def __init__(self, model, class_dims, lr, class_names=None):
        super().__init__()
        self.model = model
        self.lr = lr

        self.class_dims = class_dims
        self.class_names = class_names

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
                    average=None, num_classes=num_classes)
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

    def training_step(self, batch, batch_idx):
        x, y = batch

        output = self.forward(x)
        loss = self.step(output=output, y=y)

        self.log('train_loss',  loss, on_epoch=True, prog_bar=True)
        #self.log_all_metrics(output=output, y=y, split='train')
        
        preds = torch.argmax(output, dim=1)

        return {'loss': loss, 'preds': preds}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        output = self.forward(x)
        loss = self.step(output=output, y=y)

        self.log('valid_loss', loss, prog_bar=True)
        self.log_all_metrics(output=output, y=y, split='valid')

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        output = self.forward(x)
        loss = self.step(output=output, y=y)

        self.log('test_loss', loss, prog_bar=True)
        self.log_all_metrics(output=output, y=y, split='test',
                             on_step=False, on_epoch=True)

        return {"loss": loss}

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

                if metric_value.size():
                    # for non-averaged precision and recall, take positive class score
                    metric_value = metric_value[-1]

                log_dict[metric_key] = metric_value

            self.log_dict(log_dict, on_step=on_step, on_epoch=on_epoch)
