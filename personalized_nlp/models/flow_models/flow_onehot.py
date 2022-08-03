from typing import Dict, Any, Optional

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
        
    def _shared_step(self, batch: Dict[str, Any], y: torch.Tensor):
        annotator_ids = batch["annotator_ids"].long()

        worker_onehots = self.worker_onehots[annotator_ids]
        context = torch.cat([batch['embeddings'], worker_onehots], dim=1)
        return context, y 
        
    def sample(self, batch: Dict[str, Any], y: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        context, y = self._shared_step(batch, y)
        sample = [self.flow.sample(1, context=context).squeeze(0) for ctx in context]
        return sample
        
    def log_prob(self, batch: Dict[str, Any], y: torch.Tensor) -> torch.Tensor:
        
        # raise Exception(f"Emb: {batch['embeddings'].shape} Workers: {worker_onehots.shape} Context: {context.shape}")
        context, y = self._shared_step(batch, y)
        log_prob = self.flow.log_prob(inputs=y.float(), context=context)
        # raise Exception(f'y: {y.shape} context: {context.shape} log: {log_prob.shape}')
        return log_prob