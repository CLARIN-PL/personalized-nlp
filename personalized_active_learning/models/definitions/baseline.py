import torch.nn as nn

from personalized_active_learning.datasets import TextFeaturesBatch
from personalized_active_learning.models.interface import IModel


class Baseline(IModel):
    def __init__(
        self,
        output_dim=2,
        text_embedding_dim=768,
        **kwargs,  # TODO: eliminate kwargs / We need it to define different models
    ):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.fc1 = nn.Linear(text_embedding_dim, output_dim)

    @property
    def head(self) -> nn.Module:
        """Get the model's head.

        Returns: The model's head (I.E. last layer).

        """
        return self.fc1

    @head.setter
    def head(self, new_head: nn.Module):
        """Set the model's head."""
        self.fc1 = new_head

    def forward(self, features: TextFeaturesBatch):
        x = features.embeddings
        x = x.view(-1, self.text_embedding_dim)
        x = self.fc1(x)

        return x
