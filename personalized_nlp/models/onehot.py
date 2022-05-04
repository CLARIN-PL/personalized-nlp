import torch
import torch.nn as nn
import torch.nn.functional as F


class NetOneHot(nn.Module):
    def __init__(self, output_dim, annotator_num, text_embedding_dim=768, **kwargs):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.fc1 = nn.Linear(text_embedding_dim + annotator_num, output_dim)

        self.worker_onehots = nn.parameter.Parameter(
            torch.eye(annotator_num), requires_grad=False
        )

    def forward(self, features):
        x = features["embeddings"]
        annotator_ids = features["annotator_ids"].long()

        worker_onehots = self.worker_onehots[annotator_ids]

        x = x.view(-1, self.text_embedding_dim)
        x = self.fc1(torch.cat([x, worker_onehots], dim=1))
        return x
