# TODO: REFACTOR!!!
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import classification_report
import pytorch_lightning as pl


class PersonalizedMetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self._test_outputs: list[dict[str, Any]] = []
        self._valid_outputs: list[dict[str, Any]] = []

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._valid_outputs.append(outputs)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        annotator_ids = torch.cat(
            [o["x"].annotator_ids for o in self._valid_outputs], dim=0
        )
        y_pred = torch.cat([o["y_pred"] for o in self._valid_outputs], dim=0)
        y_true = torch.cat([o["y"] for o in self._valid_outputs], dim=0)

        self.log_all_metrics(annotator_ids, y_pred, y_true, pl_module, "valid")

        self._valid_outputs: list[dict[str, Any]] = []

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self._test_outputs.append(outputs)

    def on_test_epoch_end(self, trainer, pl_module, *args, **kwargs):
        annotator_ids = torch.cat(
            [o["x"].annotator_ids for o in self._test_outputs], dim=0
        )
        y_pred = torch.cat([o["y_pred"] for o in self._test_outputs], dim=0)
        y_true = torch.cat([o["y"] for o in self._test_outputs], dim=0)

        self.log_all_metrics(annotator_ids, y_pred, y_true, pl_module, "test")

        self._test_outputs: list[dict[str, Any]] = []

    def log_all_metrics(self, annotator_ids, output, y, pl_module, split: str):
        class_dims = pl_module.class_dims
        class_names = pl_module.class_names

        output = torch.softmax(output, dim=1)

        metrics = defaultdict(list)

        for annotator_id in np.unique(annotator_ids.cpu().numpy()):
            for cls_dim_idx in range(len(class_dims)):
                start_idx = sum(class_dims[:cls_dim_idx])
                end_idx = start_idx + class_dims[cls_dim_idx]

                person_confidences = (
                    output[annotator_ids == annotator_id, start_idx:end_idx].cpu().numpy()
                )
                person_y_pred = np.argmax(person_confidences, axis=1)
                person_y_true = (
                    y[annotator_ids == annotator_id, cls_dim_idx].long().cpu().numpy()
                )

                personal_metrics = classification_report(
                    person_y_true,
                    person_y_pred,
                    output_dict=True,
                    zero_division=0,  # We don't our output to be filled with warnings
                )

                class_name = class_names[cls_dim_idx] if class_names else str(cls_dim_idx)

                for cls_idx in range(class_dims[cls_dim_idx]):
                    if str(cls_idx) in personal_metrics:
                        value = personal_metrics[str(cls_idx)]["f1-score"]
                        metrics[f"{split}_personal_f1_{class_name}_{cls_idx}"].append(
                            value
                        )

                value = personal_metrics["macro avg"]["f1-score"]
                metrics[f"{split}_personal_macro_f1"].append(value)

        for metric_key in metrics:
            metric_values = metrics.get(metric_key)
            pl_module.log(metric_key, np.mean(metric_values))
