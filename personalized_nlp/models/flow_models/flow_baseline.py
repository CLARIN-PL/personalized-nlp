from typing import Dict, Any

import torch
from torch import nn

from personalized_nlp.models.flows import FLOWS_DICT


class FlowBaseline(nn.Module):
    
    def __init__(self, flow_type: str, flow_kwargs: Dict[str, Any], **kwargs) -> None:
        super(FlowBaseline, self).__init__()
        self.flow = FLOWS_DICT[flow_type](**flow_kwargs)
        
    def log_prob(self, batch: Dict[str, Any], y: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(inputs=y.float(), context=batch['embeddings'])
