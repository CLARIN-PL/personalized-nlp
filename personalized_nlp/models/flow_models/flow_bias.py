from typing import Dict, Any, Optional

import torch
from torch import embedding_bag, nn

from personalized_nlp.models.flows import FLOWS_DICT


class FlowBias(nn.Module):
    
    def __init__(self, 
            flow_type: str, 
            flow_kwargs: Dict[str, Any], 
            annotator_num: int,
            hidden_dim: int=100,
            output_dim: int = 100,
            **kwargs) -> None:
        super(FlowBias, self).__init__() 
        
        
        embedding_dim=flow_kwargs['context_features']
        # HuBi Simple
        self.annotator_biases = torch.nn.Embedding(
            num_embeddings=annotator_num + 1, embedding_dim=hidden_dim, padding_idx=0
        )
        self.annotator_biases.weight.data.uniform_(-0.001, 0.001)

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.softplus = nn.Softplus()
        
        # Flow
        flow_kwargs['context_features'] = output_dim
        self.flow = FLOWS_DICT[flow_type](**flow_kwargs)
        
        
    def _shared_step(self, batch: Dict[str, Any], y: torch.Tensor):
        x = batch['embeddings']
        annotator_ids = batch["annotator_ids"].long()

        x = self.fc1(x)
        x = self.softplus(x)

        annotator_bias = self.annotator_biases(annotator_ids + 1)
        
        #raise Exception(f'{x.shape} {annotator_bias.shape} {(x + annotator_bias).shape}')
        context = self.fc2(x + annotator_bias)
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