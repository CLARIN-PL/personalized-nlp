import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel

from typing import Dict, Any


class TransformerMultiUserSingleTextHead(nn.Module):
    def __init__(
        self,
        text_embedding_dim: int,
        output_dim: int,
        huggingface_model_name: str = "bert-base-cased",
        max_length: int = 128,
        append_annotator_ids=False,
        annotator_num=None,
        use_cuda=True,
        **kwargs,
    ):
        super().__init__()

        self._tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
        self._model = AutoModel.from_pretrained(huggingface_model_name)

        self.max_length = max_length
        self.use_cuda = use_cuda

        self.annotator_num = annotator_num

        self.fc1 = nn.Linear(text_embedding_dim, annotator_num * output_dim)
        # self.fc1 = nn.Linear(text_embedding_dim, output_dim)

        self.annotator_biases = torch.nn.Embedding(
            num_embeddings=annotator_num + 1, embedding_dim=output_dim, padding_idx=0
        ).to("cuda")

    def forward(self, features: Dict[str, Any]):
        texts_raw = features["raw_texts"].tolist()

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
        if self.use_cuda:
            batch_encoding = batch_encoding.to("cuda")

        output = model(**batch_encoding)

        if hasattr(output, "pooler_output"):
            emb = output.pooler_output
        else:
            emb = output.last_hidden_state[:, 0, :]

        batch_size = len(texts_raw)
        # logits = self.fc1(emb).reshape(batch_size, -1, self.annotator_num)
        logits = self.fc1(emb)[:, :, None].expand(batch_size, -1, self.annotator_num)

        # logits = (
        #     logits
        #     + self.annotator_biases(
        #         torch.arange(self.annotator_num).to("cuda") + 1
        #     ).transpose(0, 1)[None, :, :]
        # )
        return logits
