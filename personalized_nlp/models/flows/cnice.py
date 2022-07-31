from typing import Callable

import torch
from torch.nn import functional as F
from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AdditiveCouplingTransform
from nflows.transforms.normalization import BatchNorm
from nflows.distributions import StandardNormal



class cNICE(Flow):
    def __init__(
            self,
            features: int,
            hidden_features: int,
            context_features: int, 
            base_distribution=StandardNormal,
            num_layers: int = 2,
            num_blocks_per_layer: int = 2,
            activation: Callable[[torch.Tensor], torch.Tensor]=F.relu,
            dropout_probability: float = 0.0,
            batch_norm_within_layers: bool = False,
            batch_norm_between_layers: bool = False,
            **kwargs
    ):
        mask = torch.ones(features)
        mask[::2] = -1

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                context_features=context_features,  # New component
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            transform = AdditiveCouplingTransform(mask=mask, transform_net_create_fn=create_resnet)
            layers.append(transform)
            mask *= -1
            if batch_norm_between_layers:
                layers.append(BatchNorm(features=features))

        super(cNICE, self).__init__(
            transform=CompositeTransform(layers),
            distribution=base_distribution([features]),
        )