import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from personalized_nlp.settings import TRANSFORMER_MODEL_STRINGS

from typing import Dict, Any
import pytorch_lightning as pl


class NetUserID(torch.nn.Module):
    def __init__(self, output_dim, annotator_num, embedding_type, text_embedding_dim=768,
    model_name='roberta-base', max_length=256, embedding_dim=20, hidden_dim=100, frozen=False,
    **kwargs):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding_type = embedding_type

        self.fc1 = nn.Linear(text_embedding_dim, output_dim)

        additional_special_tokens = [f'_#{a_id}#_' for a_id in range(annotator_num)]
        special_tokens_dict = {'additional_special_tokens': additional_special_tokens}

        self.frozen = frozen

        if embedding_type in TRANSFORMER_MODEL_STRINGS:
            embedding_type = TRANSFORMER_MODEL_STRINGS[embedding_type]
        self.model = AutoModel.from_pretrained(embedding_type)
        self._tokenizer = AutoTokenizer.from_pretrained(embedding_type)
        self.max_length = max_length

        self._tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self._tokenizer))

    def forward(self, features):
        if not self.frozen:
            batched_texts = features['raw_texts'].tolist()
            batch_encoding = self._tokenizer.batch_encode_plus(
                batched_texts,
                padding='longest',
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            ).to(self.model.device)

            emb = self.model(**batch_encoding).pooler_output
        else:
            emb = features['embeddings']

        x = emb

        x = x.view(-1, self.text_embedding_dim)
        x = self.fc1(torch.cat([x], dim=1))
        return x
