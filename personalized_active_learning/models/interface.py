"""Interface that should be implemented by each model."""

import abc

import torch
from torch import nn
from personalized_active_learning.datasets import TextFeaturesBatch


class IModel(nn.Module, abc.ABC):
    """Interface for PyTorch models.

    Custom interface is used since we need dedicated return type instead of tensor.

    """

    @abc.abstractmethod
    def forward(self, text_features: TextFeaturesBatch) -> torch.Tensor:
        """Forward input.

        Args:
            text_features: TextFeatures from whom model must selects what it needs.

        Returns:
            tensor that can be processed by further layers or used as output.

        """
