import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from typing import Dict, Any


class TransformerUserId(nn.Module):
    def __init__(
        self,
        text_embedding_dim: int,
        output_dim: int,
        huggingface_model_name: str = "bert-base-cased",
        max_length: int = 256,
        append_annotator_ids=False,
        annotator_num=None,
        **kwargs,
    ):
        super().__init__()

        self.append_annotator_ids = append_annotator_ids
        if append_annotator_ids:
            additional_special_tokens = [f"_#{a_id}#_" for a_id in range(annotator_num)]
            special_tokens_dict = {
                "additional_special_tokens": additional_special_tokens
            }

            self._tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
            self._tokenizer.add_special_tokens(special_tokens_dict)

            self._model = AutoModel.from_pretrained(huggingface_model_name)
            self._model.resize_token_embeddings(len(self._tokenizer))
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
            self._model = AutoModel.from_pretrained(huggingface_model_name)

        self.max_length = max_length

        self.fc1 = nn.Linear(text_embedding_dim, output_dim)

    def forward(self, features: Dict[str, Any]):
        texts_raw = features["raw_texts"].tolist()
        annotator_ids = features["annotator_ids"].tolist()

        if self.append_annotator_ids:
            texts_raw = [
                f"_#{a_id}#_ " + t for t, a_id in zip(texts_raw, annotator_ids)
            ]

        tokenizer = self._tokenizer
        model = self._model

        batch_encoding = tokenizer.batch_encode_plus(
            texts_raw,
            padding="longest",
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_encoding = batch_encoding.to("cuda")

        emb = model(**batch_encoding).pooler_output

        return self.fc1(emb)
