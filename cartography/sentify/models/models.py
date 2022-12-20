import abc
from time import time
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import Tensor, nn
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, F1Score
from transformers import AdamW, PreTrainedModel, get_linear_schedule_with_warmup

from sentify.models.retriever import Retriever


class BaseSentimentModel(pl.LightningModule, abc.ABC):
    def __init__(
        self,
        learning_rate: float = 1e-5,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        num_classes: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        self._loss_func = nn.CrossEntropyLoss()
        self._num_classes = num_classes

        metrics = MetricCollection(
            {
                'accuracy': Accuracy(average='micro'),
                'precision': Precision(average='micro'),
                'recall': Recall(average='micro'),
                'f1_micro': F1Score(average='micro'),
                'f1_macro': F1Score(average='macro', num_classes=num_classes),
                'f1_score_per_class': F1Score(average='none', num_classes=num_classes),
            }
        )
        self._train_metrics = metrics.clone(prefix='train/')
        self._val_metrics = metrics.clone(prefix='val/')
        self._test_metrics = metrics.clone(prefix='test/')

    @abc.abstractmethod
    def forward(  # type: ignore
        self,
        *args,
        **kwargs,
    ):
        pass

    @abc.abstractmethod
    def _step(
        self,
        batch: Dict[str, Tensor],
        step_type: str,
        batch_idx: int,
    ):
        pass

    def _common_step(
        self,
        batch: Dict[str, Tensor],
        step_type: str,
        batch_idx: int,
    ):
        output = self._step(batch, step_type, batch_idx)
        labels = batch['labels']
        return {
            **output,
            'labels': labels,
        }

    def training_step(  # type: ignore
        self,
        batch: Dict[str, Tensor],
        batch_idx: int,
    ) -> Dict[str, Tensor]:
        output = self._common_step(batch, 'train', batch_idx)  # type: ignore
        return {
            'loss': output['loss'],
            'labels': output['labels'],
            'logits': output['logits'].detach(),
            'guids': batch['guids'].detach(),
        }

    def validation_step(  # type: ignore
        self,
        batch: Dict[str, Tensor],
        batch_idx: int,
    ):
        output = self._common_step(batch, 'val', batch_idx)  # type: ignore
        return {
            'logits': output['logits'],
            'labels': output['labels'],
        }

    def test_step(  # type: ignore
        self,
        batch: Dict[str, Tensor],
        batch_idx: int,
    ):
        output = self._common_step(batch, 'test', batch_idx)  # type: ignore
        return {
            'logits': output['logits'],
            'labels': output['labels'],
        }

    def on_train_epoch_start(self) -> None:
        self._epoch_start_time = time()

    def training_epoch_end(self, outputs) -> None:
        epoch_time = time() - self._epoch_start_time
        self.log('train/epoch_time', epoch_time, on_epoch=True, on_step=False)
        self._epoch_end(outputs, metric_obj=self._train_metrics)

    def validation_epoch_end(self, outputs) -> None:
        self._epoch_end(outputs, metric_obj=self._val_metrics)

    def test_epoch_end(self, outputs) -> None:
        self._epoch_end(outputs, metric_obj=self._test_metrics)

    def _epoch_end(self, outputs, metric_obj: torchmetrics.MetricCollection) -> None:
        logits = torch.cat([out['logits'] for out in outputs]).float()
        labels = torch.cat([out['labels'] for out in outputs]).int()

        metric_dict = metric_obj(logits, labels)
        step_type = metric_obj.prefix

        f1_class_key = f'{step_type}f1_score_per_class'
        labels_str = [f'{f1_class_key}/{label_idx}' for label_idx in range(self._num_classes)]
        f1_class = metric_dict.pop(f1_class_key)
        metrics_per_class = dict(zip(labels_str, f1_class))

        metrics = metric_dict | metrics_per_class
        self.log_dict(metrics, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            params=optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=self.adam_epsilon,
            weight_decay=self.hparams.weight_decay,
        )

        if self.warmup_steps > 0:
            # calculate total learning steps
            # call len(self.train_dataloader()) should be fixed in pytorch-lightning v1.6
            self._total_train_steps = (
                self.trainer.max_epochs
                * len(self.trainer._data_connector._train_dataloader_source.dataloader())
                * self.trainer.accumulate_grad_batches
            )

            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self._total_train_steps,
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                },
            }
        else:
            return {'optimizer': optimizer}


class TransformerSentimentModel(BaseSentimentModel):
    def __init__(
        self,
        model: PreTrainedModel,
        learning_rate: float = 1e-5,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        num_classes: int = 3,
    ):
        super().__init__(
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            num_classes=num_classes,
        )
        self.model = model

    def forward(  # type: ignore
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def _step(
        self,
        batch: Dict[str, Tensor],
        step_type: str,
        batch_idx: int,
    ):
        output = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )

        logits, labels = output.logits, batch['labels']

        output['loss'] = self._loss_func(logits, labels)
        self.log(f'{step_type}/loss', output.loss, on_epoch=True, on_step=False)
        return output


class RetrieverModel(BaseSentimentModel):
    def __init__(
        self,
        model: Retriever,
        learning_rate: float = 1e-5,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        num_classes: int = 3,
    ):
        super().__init__(
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            num_classes=num_classes,
        )
        self.model = model

    def forward(  # type: ignore
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        user_texts_similarities: Tensor,
        user_texts_labels: Tensor,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            user_texts_similarities=user_texts_similarities,
            user_texts_labels=user_texts_labels,
        )

    def _step(
        self,
        batch: Dict[str, Tensor],
        step_type: str,
        batch_idx: int,
    ):
        logits = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            user_texts_similarities=batch['user_texts_similarities'],
            user_texts_labels=batch['user_texts_labels'],
        )

        labels = batch['labels']

        loss = self._loss_func(logits, labels)
        self.log(
            f'{step_type}/loss',
            loss,
            on_epoch=True,
            on_step=False,
        )
        return {
            'loss': loss,
            'logits': logits,
        }
