import abc
import numpy as np

from personalized_active_learning.datamodules import BaseDataModule
from personalized_active_learning.models import IModel


class ISelfSupervisedTrainer(abc.ABC):
    """Perform self-supervised training."""

    @abc.abstractmethod
    def train(
            self,
            dataset: BaseDataModule,
            model: IModel,
            pseudo_labels: np.ndarray,
            unlabelled_indexes: np.ndarray,
    ) -> IModel:
        """Use self-supervised learnng to train model.

        Args:
            dataset: Dataset containing data used for pretraining.
            model: Model that will be trained.
            pseudo_labels: Labels created from unlabelled examples.
            unlabelled_indexes: Indexes of unlabelled examles.

        Returns:
            Trained model.

        """
        raise NotImplementedError
