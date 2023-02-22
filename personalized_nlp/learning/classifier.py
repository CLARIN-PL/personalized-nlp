from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from torchmetrics.classification.f_beta import F1Score
from torchmetrics.classification.accuracy import Accuracy
from personalized_nlp.utils.metrics import F1Class, PrecisionClass, RecallClass


class Classifier(pl.LightningModule):
    def __init__(
        self,
        model,
        class_dims,
        lr: float,
        class_names=None,
        log_valid_metrics=False,
        ignore_index=-1,
    ) -> None:

        super(Classifier, self).__init__()
        self.save_hyperparameters()

        self.model = model
        self.lr = lr

        self.class_dims = class_dims
        self.class_names = class_names

        self.metric_types = ("accuracy", "precision", "recall", "f1", "macro_f1")
        self.log_valid_metrics = log_valid_metrics
        self.ignore_index = ignore_index

        class_metrics = {}

        for split in ["train", "valid", "test"]:
            for class_idx in range(len(class_dims)):
                class_name = class_names[class_idx] if class_names else str(class_idx)
                num_classes = class_dims[class_idx]

                class_metrics[f"{split}_accuracy_{class_name}"] = Accuracy(
                    task="multiclass", num_classes=num_classes, ignore_index=self.ignore_index
                )

                for class_dim in range(num_classes):
                    class_metrics[
                        f"{split}_precision_{class_name}_{class_dim}"
                    ] = PrecisionClass(
                        task='multiclass',
                        num_classes=num_classes,
                        average=None,
                        class_idx=class_dim,
                        ignore_index=self.ignore_index,
                    )
                    class_metrics[
                        f"{split}_recall_{class_name}_{class_dim}"
                    ] = RecallClass(
                        task='multiclass',
                        num_classes=num_classes,
                        average=None,
                        class_idx=class_dim,
                        ignore_index=self.ignore_index,
                    )
                    class_metrics[f"{split}_f1_{class_name}_{class_dim}"] = F1Class(
                        task='multiclass',
                        average="none",
                        num_classes=num_classes,
                        class_idx=class_dim,
                        ignore_index=self.ignore_index,
                    )
                class_metrics[f"{split}_macro_f1_{class_name}"] = F1Score(
                    task='multiclass',
                    average="macro",
                    num_classes=num_classes,
                    ignore_index=self.ignore_index,
                )

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

            loss = loss + nn.CrossEntropyLoss(ignore_index=self.ignore_index)(
                output[:, start_idx:end_idx], y[:, cls_idx].long()
            )

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        batch_size = y.size()[0]

        output = self.forward(x)
        loss = self.step(output=output, y=y)

        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        preds = torch.argmax(output, dim=1)

        return {
            "loss": loss,
            "output": output,
            "preds": preds,
            "y": y,
            "x": x,
            "class_names": self.class_names,
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = y.size()[0]

        output = self.forward(x)
        loss = self.step(output=output, y=y)

        self.log("valid_loss", loss, prog_bar=True, batch_size=batch_size)
        if self.log_valid_metrics:
            self.log_all_metrics(output=output, y=y, split="valid")

        return {
            "loss": loss,
            "output": output,
            "y": y,
            "x": x,
            "y_pred": output,
            "is_regression": False,
            "class_names": self.class_names,
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        batch_size = y.size()[0]

        output = self.forward(x)
        loss = self.step(output=output, y=y)

        self.log("test_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log_all_metrics(
            output=output,
            y=y,
            split="test",
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return {
            "loss": loss,
            "output": output,
            "y": y,
            "x": x,
            "y_pred": output,
            "is_regression": False,
            "class_names": self.class_names,
        }

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, _ = batch

        return {"output": self.forward(x)}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_all_metrics(
        self, output, y, split, on_step=None, on_epoch=None, batch_size=None
    ):
        class_dims = self.class_dims
        class_names = self.class_names

        output = torch.softmax(output, dim=1)

        for cls_idx in range(len(class_dims)):
            start_idx = sum(class_dims[:cls_idx])
            end_idx = start_idx + class_dims[cls_idx]

            class_name = class_names[cls_idx] if class_names else str(cls_idx)

            log_dict = {}
            for metric_type in self.metric_types:
                metric_key_prefix = f"{split}_{metric_type}_{class_name}"

                metric_keys = [
                    k for k in self.metrics.keys() if k.startswith(metric_key_prefix)
                ]

                class_output = output[:, start_idx:end_idx].float()
                class_y = y[:, cls_idx].long()

                for metric_key in metric_keys:
                    self.metrics[metric_key].update(class_output, class_y)

                    log_dict[metric_key] = self.metrics[metric_key]

                # if split == "valid" and metric_type == "macro_f1":
                #     f1_macros = [
                #         log_dict[metric_key].compute().cpu()
                #         for metric_key in metric_keys
                #     ]

                #     log_dict["valid_macro_f1_mean"] = np.mean(f1_macros)

            self.log_dict(
                log_dict,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=True,
                batch_size=batch_size,
            )

    def decode_predictions(self, probabs: torch.Tensor) -> torch.Tensor:
        class_dims = self.class_dims
        predictions = []
        for class_idx, _ in enumerate(class_dims):
            start_idx = sum(class_dims[:class_idx])
            end_idx = start_idx + class_dims[class_idx]

            class_predictions = probabs[:, start_idx:end_idx].argmax(dim=1)
            predictions.append(class_predictions.unsqueeze(1))

        return torch.cat(predictions, dim=1)
