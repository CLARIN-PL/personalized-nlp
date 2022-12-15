import abc
import numpy as np

from typing import Optional
from personalized_active_learning.datamodules import BaseDataModule
from personalized_active_learning.models import IModel


class ILabelPropagator(abc.ABC):
    """Perform label propagation."""

    @abc.abstractmethod
    def propagate_labels(
        self,
        dataset: BaseDataModule,
        model: Optional[IModel] = None  # TODO How passing and AL Flow work together?
    ) -> np.ndarray:
        """Use label propagation to get new pseudo labels.

        Args:
            dataset: Dataset containing data used for pretraining.
            model: Model that will be used for label propagation
                (Can be none as we sometimes create such model inside).

        Returns:
            Pseudo-labels.

        """
        raise NotImplementedError
