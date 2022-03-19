import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from typing import Dict, Any

class NetUserID(nn.Module):
    def __init__(self, output_dim, annotator_num, model_name='roberta-base', max_length=128, text_embedding_dim=768, **kwargs):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.fc1 = nn.Linear(text_embedding_dim, output_dim)
        
        additional_special_tokens = [f'_#{a_id}#_' for a_id in range(annotator_num)]
        special_tokens_dict = {'additional_special_tokens': additional_special_tokens}

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.add_special_tokens(special_tokens_dict)

        self._model = AutoModel.from_pretrained(model_name)
        self._model.resize_token_embeddings(len(self._tokenizer))

        self.max_length = max_length

    def forward(self, features: Dict[str, Any]):
        texts_raw = features['embeddings'].tolist()
        
        tokenizer = self._tokenizer
        model = self._model
        
        batch_encoding = tokenizer.batch_encode_plus(
            texts_raw,
            padding='longest',
            add_special_tokens=True,
            truncation=True, max_length=self.max_length,
            return_tensors='pt',
        )
        batch_encoding = batch_encoding.to('cuda')

        emb = model(**batch_encoding).pooler_output
        
        return emb
