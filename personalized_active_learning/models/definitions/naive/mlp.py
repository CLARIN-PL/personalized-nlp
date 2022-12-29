from collections import OrderedDict
from typing import Optional, List

import torch.nn as nn

from personalized_active_learning.datamodules import TextFeaturesBatch
from personalized_active_learning.models.interface import IModel

try:
    from itertools import pairwise
except ImportError:
    from personalized_active_learning.integration_tools import pairwise


class Mlp(IModel):
    def __init__(
        self,
        output_dim: int = 2,
        text_embedding_dim: int = 768,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        hidden_sizes = hidden_sizes or [text_embedding_dim]
        self.input_layer = nn.Linear(text_embedding_dim, hidden_sizes[0])
        self.input_relu = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_sizes[-1], output_dim)
        hidden_layers = []
        for index, (first_hidden_size, second_hidden_size) in enumerate(
            pairwise(hidden_sizes)
        ):
            hidden_layers.append(
                (
                    f"hidden_linear_{index}",
                    nn.Linear(first_hidden_size, second_hidden_size),
                )
            )
            hidden_layers.append((f"hidden_relu_{index}", nn.ReLU()))
            hidden_layers.append((f"hidden_dropout_{index}", nn.Dropout(dropout)))
        self.hidden_layers = nn.Sequential(
            OrderedDict(
                hidden_layers,
            )
        )

    def get_head(self) -> nn.Module:
        """Get the model's head.

        Returns: The model's head (I.E. last layer).

        """
        return self.output_layer

    def set_head(self, new_head: nn.Module):
        """Set the model's head.

        Args:
            new_head: The new model's head.

        """
        self.output_layer = new_head

    def forward(self, features: TextFeaturesBatch):
        x = features.embeddings
        x = x.view(-1, self.text_embedding_dim)
        x = self.input_layer(x)
        x = self.input_relu(x)
        x = self.input_dropout(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
