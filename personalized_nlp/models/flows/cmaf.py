from typing import Callable

import torch
from torch.nn import functional as F
from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import RandomPermutation, ReversePermutation
from nflows.transforms.normalization import BatchNorm
from nflows.distributions import StandardNormal



class cMAF(Flow):
    def __init__(
            self,
            features: int,
            hidden_features: int,
            context_features: int, 
            base_distribution=StandardNormal,
            num_layers: int = 2,
            num_blocks_per_layer: int = 2,
            use_random_permutations: bool = False,
            use_residual_blocks: bool = False,
            use_random_masks: bool = True,
            activation: Callable[[torch.Tensor], torch.Tensor]=F.relu,
            dropout_probability: float = 0.0,
            batch_norm_within_layers: bool = False,
            batch_norm_between_layers: bool = False,
            **kwargs
    ):


        if use_random_permutations:
            permutation_constructor = RandomPermutation
        else:
            permutation_constructor = ReversePermutation

        layers = []
        for _ in range(num_layers):
            layers.append(permutation_constructor(features))
            layers.append(
                MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks_per_layer,
                    context_features=context_features,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=use_random_masks,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
                )
            )
            if batch_norm_between_layers:
                layers.append(BatchNorm(features=features))

        super(cMAF, self).__init__(
            transform=CompositeTransform(layers),
            distribution=base_distribution([features]),
        )