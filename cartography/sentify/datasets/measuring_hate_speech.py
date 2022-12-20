from abc import abstractmethod, ABCMeta, ABC
from pathlib import Path
from typing import Optional

import pandas as pd
from pytorch_lightning import seed_everything
from transformers import PreTrainedTokenizerFast

from sentify import DATASETS_PATH
from sentify.datasets.base_datamodule import AugmentationType, BaseDataModule
from sentify.datasets.dataset import SampleType


class MeasuringHateSpeechDataModule(BaseDataModule, metaclass=ABCMeta):
    SPLIT_MAPPER = {
        'train': 1,
        'val': 2,
        'test': 3,
    }

    def __init__(
            self,
            tokenizer: PreTrainedTokenizerFast,
            dataset_path: Path = DATASETS_PATH.joinpath('MHS'),
            augmentations: Optional[AugmentationType] = None,
            batch_size: int = 16,
            num_workers: int = 8,
            seed: int = 2718,
            max_length: int = int(1e10),
    ):
        super().__init__(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            augmentations=augmentations,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
        )
        self.dimension = None

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        seed_everything(self._seed)
        self._train_samples = self._load_dataset(self._dataset_path.joinpath('train.tsv'))
        self._val_samples = self._load_dataset(self._dataset_path.joinpath('dev.tsv'))
        self._test_samples = self._load_dataset(self._dataset_path.joinpath('test.tsv'))

    def _load_dataset(self, path: Path) -> list[SampleType]:
        df = pd.read_csv(
            path,
            sep='\t',
            engine='python',
            dtype={
                'index': int,
            },
        )
        df = df.rename(columns={self.dimension: 'label'})
        return df.to_dict(orient='records')

    def _map_index_to_int(self, index: str) -> int:
        split, idx = index.split('_')
        split_id = self.SPLIT_MAPPER[split]
        new_index = f'{split_id}{idx}'
        return int(new_index)


class SentimentMHSDataModule(MeasuringHateSpeechDataModule):

    def __init__(
            self,
            tokenizer: PreTrainedTokenizerFast,
            dataset_path: Path = DATASETS_PATH.joinpath('MHS'),
            augmentations: Optional[AugmentationType] = None,
            batch_size: int = 16,
            num_workers: int = 8,
            seed: int = 2718,
            max_length: int = int(1e10),
    ):
        super().__init__(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            augmentations=augmentations,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
        )
        self.dimension = 'sentiment'

    @property
    def name(self) -> str:
        return 'MHS_sentiment'

    @property
    def num_classes(self) -> int:
        return 5


class HateSpeechMHSDataModule(MeasuringHateSpeechDataModule):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerFast,
            dataset_path: Path = DATASETS_PATH.joinpath('MHS'),
            augmentations: Optional[AugmentationType] = None,
            batch_size: int = 16,
            num_workers: int = 8,
            seed: int = 2718,
            max_length: int = int(1e10),
    ):
        super().__init__(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            augmentations=augmentations,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
        )
        self.dimension = 'hatespeech'

    @property
    def name(self) -> str:
        return 'MHS_hatespeech'

    @property
    def num_classes(self) -> int:
        return 3


class InsultMHSDataModule(MeasuringHateSpeechDataModule):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerFast,
            dataset_path: Path = DATASETS_PATH.joinpath('MHS'),
            augmentations: Optional[AugmentationType] = None,
            batch_size: int = 16,
            num_workers: int = 8,
            seed: int = 2718,
            max_length: int = int(1e10),
    ):
        super().__init__(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            augmentations=augmentations,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
        )
        self.dimension = 'insult'

    @property
    def name(self) -> str:
        return 'MHS_insult'

    @property
    def num_classes(self) -> int:
        return 5


class ViolenceMHSDataModule(MeasuringHateSpeechDataModule):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerFast,
            dataset_path: Path = DATASETS_PATH.joinpath('MHS'),
            augmentations: Optional[AugmentationType] = None,
            batch_size: int = 16,
            num_workers: int = 8,
            seed: int = 2718,
            max_length: int = int(1e10),
    ):
        super().__init__(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            augmentations=augmentations,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
        )
        self.dimension = 'violence'

    @property
    def name(self) -> str:
        return 'MHS_violence'

    @property
    def num_classes(self) -> int:
        return 5


class HumiliateMHSDataModule(MeasuringHateSpeechDataModule):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerFast,
            dataset_path: Path = DATASETS_PATH.joinpath('MHS'),
            augmentations: Optional[AugmentationType] = None,
            batch_size: int = 16,
            num_workers: int = 8,
            seed: int = 2718,
            max_length: int = int(1e10),
    ):
        super().__init__(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            augmentations=augmentations,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
        )
        self.dimension = 'humiliate'

    @property
    def name(self) -> str:
        return 'MHS_humiliate'

    @property
    def num_classes(self) -> int:
        return 5
