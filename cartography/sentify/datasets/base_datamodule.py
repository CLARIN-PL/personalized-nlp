import abc
from itertools import chain, repeat
from pathlib import Path
from typing import Any, Optional, Iterable

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, DataCollatorWithPadding

from sentify.datasets.dataset import AugmentationType, SentimentDataset, SampleType


class BaseDataModule(LightningDataModule, abc.ABC):
    COLLATE_MAPPING = {
        'index': 'guids',
        'label': 'labels',
    }

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        dataset_path: Optional[Path] = None,
        augmentations: Optional[AugmentationType] = None,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
    ):
        super().__init__()
        self.augmentations = augmentations
        self._dataset_path = dataset_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._seed = seed
        self._max_length = max_length
        self._tokenizer = tokenizer
        self._train_samples: Optional[dict, Any] = None
        self._val_samples = None
        self._test_samples = None

        self.collate_fn = DataCollatorWithPadding(self._tokenizer)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._create_dataloader(self._train_samples, shuffle=True, split='train')

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._val_samples, shuffle=False, split='val')

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._test_samples, shuffle=False, split='test')

    @property
    def samples(self) -> Iterable[tuple[SampleType, str]]:
        yield from chain(
            zip(self._train_samples, repeat('train')),
            zip(self._val_samples, repeat('val')),
            zip(self._test_samples, repeat('test')),
        )

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        pass

    def collate(
        self,
        features: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        example = features[0]
        if 'user_texts_labels' in example:  # retriever mode
            max_size = max((len(feature['user_texts_labels']) for feature in features), default=0)

            user_texts_labels = []
            user_texts_similarities = []
            for feature in features:
                diff = max_size - len(feature['user_texts_labels'])
                user_texts_labels.append(
                    F.pad(
                        input=feature.pop('user_texts_labels'),
                        pad=(0, diff),
                        mode='constant',
                        value=0,
                    ),
                )
                user_texts_similarities.append(
                    F.pad(
                        input=feature.pop('user_texts_similarities'),
                        pad=(0, diff),
                        mode='constant',
                        value=0,
                    ),
                )
            collated_features = self.collate_fn(features)
            collated_features = self._collate_mapping(collated_features)

            return {
                **collated_features,
                'user_texts_labels': torch.stack(user_texts_labels),
                'user_texts_similarities': torch.stack(user_texts_similarities),
            }
        else:
            collated_features = self.collate_fn(features)
            collated_features = self._collate_mapping(collated_features)

        return collated_features

    def _create_dataloader(
        self,
        samples: list[dict[str, Any]],
        shuffle: bool,
        split: str,
    ) -> DataLoader:
        return DataLoader(
            dataset=SentimentDataset(
                samples=samples,
                tokenizer=self._tokenizer,
                split=split,
                max_length=self._max_length,
                augmentations=self.augmentations,
            ),
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            collate_fn=self.collate,
            shuffle=shuffle,
        )

    def _collate_mapping(
        self,
        features: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        for old, new in self.COLLATE_MAPPING.items():
            if old in features:
                if new in features:
                    raise ValueError(f"Feature {new} already exists.")
                features[new] = features.pop(old)
        return features
