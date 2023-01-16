import torch
import torch.nn as nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
import pytorch_lightning as pl


class Regressor(pl.LightningModule):
    def __init__(
        self, model, lr, class_names, log_valid_metrics: bool = False, ignore_index=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.log_valid_metrics = log_valid_metrics
        self.class_names = class_names

        self.metric_types = ["r2"]

        class_metrics = {}

        for split in ["train", "valid", "test"]:
            for class_idx in range(len(class_names)):
                class_name = class_names[class_idx]

                class_metrics[f"{split}_mae_{class_name}"] = MeanAbsoluteError()
                class_metrics[f"{split}_mse_{class_name}"] = MeanSquaredError()
                class_metrics[f"{split}_r2_{class_name}"] = R2Score()

            class_metrics[f"{split}_mae_mean"] = MeanAbsoluteError()
            class_metrics[f"{split}_mse_mean"] = MeanSquaredError()
            class_metrics[f"{split}_r2_mean"] = R2Score(
                num_outputs=len(self.class_names), multioutput="uniform_average"
            )

        self.ignore_index = ignore_index
        self.metrics = nn.ModuleDict(class_metrics)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()

        output = self.forward(x)

        if self.ignore_index is not None:
            output = output[y != self.ignore_index]
            y = y[y != self.ignore_index]

        loss = nn.MSELoss()(output.float(), y.float())

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def training_epoch_end(self, outs):
        pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()

        output = self.forward(x)

        loss = nn.MSELoss()(output, y)

        self.log("valid_loss", loss, prog_bar=True)

        if self.log_valid_metrics:
            self.log_all_metrics(output=output, y=y, split="valid")

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()

        output = self.forward(x)

        loss = nn.MSELoss()(output, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log_all_metrics(
            output=output, y=y, split="test", on_step=False, on_epoch=True
        )

        return {
            "loss": loss,
            "output": output,
            "y": y,
            "x": x,
            "y_pred": output,
            "is_regression": True,
            "class_names": self.class_names,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_all_metrics(self, output, y, split, on_step=None, on_epoch=None):
        class_names = self.class_names

        log_dict = {}
        for metric_type in self.metric_types:
            metric_values = []
            for class_idx, class_name in enumerate(class_names):
                class_output = output[:, class_idx]
                class_y = y[:, class_idx]

                if self.ignore_index is not None:
                    class_output = class_output[class_y != self.ignore_index]
                    class_y = class_y[class_y != self.ignore_index]

                metric_key = f"{split}_{metric_type}_{class_name}"
                metric_value = self.metrics[metric_key].update(class_output, class_y)

                metric_values.append(metric_value)
                log_dict[metric_key] = self.metrics[metric_key]

            mean_metric_key = f"{split}_{metric_type}_mean"

            self.metrics[mean_metric_key].update(output, y)
            log_dict[mean_metric_key] = self.metrics[mean_metric_key]

            self.log_dict(log_dict, on_step=on_step, on_epoch=on_epoch)
