from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import Callback
from sklearn.metrics import classification_report


class PersonalizedMetricsCallback(Callback):
    """Class responsible for computing personalized metrics."""    
    def __init__(self):
        super().__init__()
        self._test_outputs: List[Dict[str, Any]] = []
        self._valid_outputs: List[Dict[str, Any]] = []


    def on_validation_batch_end(
        self: "PersonalizedMetricsCallback",
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Gather model's outputs for validation batch.

        Args:
            trainer (pl.Trainer): Model trainer.
            pl_module (pl.LightningModule): Trained model.
            outputs (Dict[str, Any]): Model outputs for current validation batch.
            batch (Any): Current validation batch.
            batch_idx (int): Current validation batch index.
            dataloader_idx (int): Dataloader index.
        """        
        self._valid_outputs.append(outputs)


    def on_validation_epoch_end(
        self: "PersonalizedMetricsCallback",
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule"
    ) -> None:
        """Compute personalized metrics at the end of a validation epoch.

        Args:
            trainer (pl.Trainer): Model trainer.
            pl_module (pl.LightningModule): Trained model.
        """        
        annotator_ids = torch.cat(
            [o["x"].annotator_ids for o in self._valid_outputs], dim=0
        )

        y_pred = torch.cat([o["y_pred"] for o in self._valid_outputs], dim=0)
        y_true = torch.cat([o["y"] for o in self._valid_outputs], dim=0)

        self.log_all_metrics(
            annotator_ids=annotator_ids,
            y_pred=y_pred,
            y_true=y_true,
            pl_module=pl_module,
            split="valid"
        )

        self._valid_outputs: List[Dict[str, Any]] = []


    def on_test_batch_end(
        self: "PersonalizedMetricsCallback",
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Gather model outputs for test batch.

        Args:
            trainer (pl.Trainer): Model trainer.
            pl_module (pl.LightningModule): Trained model.
            outputs (Dict[str, Any]): Model ouputs for current test batch.
        """        
        self._test_outputs.append(outputs)


    def on_test_epoch_end(
        self: "PersonalizedMetricsCallback",
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        *args: Any,
        **kwargs: Any
        ) -> None:
        """Compute personalized metrics at the end of a test epoch.

        Args:
            trainer (pl.Trainer): Model trainer.
            pl_module (pl.LightningModule): Trained model.
        """        
        annotator_ids = torch.cat(
            [o["x"].annotator_ids for o in self._test_outputs], dim=0
        )

        y_pred = torch.cat([o["y_pred"] for o in self._test_outputs], dim=0)
        y_true = torch.cat([o["y"] for o in self._test_outputs], dim=0)

        self.log_all_metrics(
            annotator_ids=annotator_ids,
            y_pred=y_pred,
            y_true=y_true,
            pl_module=pl_module,
            split="test",
        )

        self._test_outputs: List[Dict[str, Any]] = []


    def log_all_metrics(
        self: "PersonalizedMetricsCallback",
        annotator_ids: "torch.Tensor",
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        pl_module: "pl.LightningModule",
        split: str
    ) -> None:
        """Log all personalized metrics for model.

        Args:
            annotator_ids (torch.Tensor): Tensor with annotator ids.
            y_pred (torch.Tensor): Model predictions.
            y_true (torch.Tensor): Ground truth.
            pl_module (pl.LightningModule): Trained model.
            split (str): Type of split.
        """        
        class_dims = pl_module.class_dims
        class_names = pl_module.class_names

        y_pred = torch.softmax(y_pred, dim=1)

        metrics = defaultdict(list)

        for annotator_id in np.unique(annotator_ids.cpu().numpy()):
            for cls_dim_idx in range(len(class_dims)):
                start_idx = sum(class_dims[:cls_dim_idx])
                end_idx = start_idx + class_dims[cls_dim_idx]

                person_confidences = (
                    y_pred[annotator_ids == annotator_id, start_idx:end_idx].cpu().numpy()
                )

                person_y_pred = np.argmax(person_confidences, axis=1)
                person_y_true = (
                    y_true[annotator_ids == annotator_id, cls_dim_idx].long().cpu().numpy()
                )

                personal_metrics = classification_report(
                    y_true=person_y_true,
                    y_pred=person_y_pred,
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
            pl_module.log(name=metric_key, value=np.mean(metric_values))
