from typing import Dict, Any

import torch
from torch import nn

from personalized_nlp.models.flows import FLOWS_DICT


class FlowOneHot(nn.Module):
    
    def __init__(self, flow_type: str, flow_kwargs: Dict[str, Any], annotator_num: int, **kwargs) -> None:
        super(FlowOneHot, self).__init__()
        
        flow_kwargs['context_features'] = flow_kwargs['context_features'] + annotator_num
        
        self.flow = FLOWS_DICT[flow_type](**flow_kwargs)
        
        self.worker_onehots = nn.parameter.Parameter(
            torch.eye(annotator_num), requires_grad=False
        )
        
    def log_prob(self, batch: Dict[str, Any], y: torch.Tensor) -> torch.Tensor:
        annotator_ids = batch["annotator_ids"].long()

        worker_onehots = self.worker_onehots[annotator_ids]
        context = torch.cat([batch['embeddings'], worker_onehots], dim=1)
        # raise Exception(f"Emb: {batch['embeddings'].shape} Workers: {worker_onehots.shape} Context: {context.shape}")
        return self.flow.log_prob(inputs=y.float(), context=context)
