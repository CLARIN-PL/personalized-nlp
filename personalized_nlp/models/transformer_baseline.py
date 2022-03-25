import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from personalized_nlp.settings import TRANSFORMER_MODEL_STRINGS


class TransformerBaseline(nn.Module):
    def __init__(self, output_dim, finetune: bool = True,
                 model_name='roberta-base', max_length=128, text_embedding_dim=768, **kwargs):
        super().__init__()

        self.finetune = finetune

        if self.finetune:
            if model_name in TRANSFORMER_MODEL_STRINGS:
                model_name = TRANSFORMER_MODEL_STRINGS[model_name]

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            self.max_length = max_length

        self.text_embedding_dim = text_embedding_dim
        self.fc1 = nn.Linear(self.text_embedding_dim, output_dim)

    def forward(self, features):
        if self.finetune:
            texts_raw = features['raw_texts'].tolist()
            tokenizer = self._tokenizer
            model = self._model

            batch_encoding = tokenizer.batch_encode_plus(
                texts_raw,
                padding='longest',
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
            batch_encoding = batch_encoding.to(model.device)
            emb = model(**batch_encoding).pooler_output
        else:
            emb = features['embeddings']

        x = emb
        x = x.view(-1, self.text_embedding_dim)

        x = self.fc1(x)
        return x
