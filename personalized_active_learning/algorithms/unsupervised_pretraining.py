import abc
from typing import Optional

import pytorch_lightning as pl

from personalized_active_learning.datasets import BaseDataset
from personalized_active_learning.models import IModel


class IUnsupervisedPretrainer(abc.ABC):
    """Perform unsupervised pre-training.

    As defined in paper "Rethinking deep active learning:
    Using unlabeled data at model training".

    """

    @abc.abstractmethod
    def pretrain(
        self,
        dataset: BaseDataset,
        model: IModel,
        random_state: Optional[int] = None,
    ) -> pl.Trainer:
        """

        Args:
            dataset: Dataset containing data used for pretraining.
            model: Model that will be pretrained.
            random_state: Random state.

        Returns:
            Trainer used to train model. Can be used to load model weights.

        """
        raise NotImplementedError
