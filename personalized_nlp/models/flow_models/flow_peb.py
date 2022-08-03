from typing import Dict, Any, Optional

import torch
from torch import nn

from personalized_nlp.models.flows import FLOWS_DICT


class FlowPEB(nn.Module):
    
    def __init__(self, flow_type: str, flow_kwargs: Dict[str, Any], bias_vector_length: int, **kwargs) -> None:
        super(FlowPEB, self).__init__()
        
        flow_kwargs['context_features'] = flow_kwargs['context_features'] + bias_vector_length
        
        self.flow = FLOWS_DICT[flow_type](**flow_kwargs)
        
        
    def _shared_step(self, batch: Dict[str, Any], y: torch.Tensor):
        context = torch.cat([batch['embeddings'], batch["annotator_biases"].float()], dim=1)
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