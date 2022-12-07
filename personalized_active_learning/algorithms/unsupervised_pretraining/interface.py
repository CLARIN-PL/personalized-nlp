import abc

from personalized_active_learning.datasets import BaseDataModule
from personalized_active_learning.models import IModel


class IUnsupervisedPretrainer(abc.ABC):
    """Perform unsupervised pre-training."""

    @abc.abstractmethod
    def pretrain(
        self,
        dataset: BaseDataModule,
        model: IModel,
    ) -> IModel:
        """Pretrain model.

        Args:
            dataset: Dataset containing data used for pretraining.
            model: Model that will be pretrained.

        Returns:
            Pretrained model.

        """
        raise NotImplementedError
