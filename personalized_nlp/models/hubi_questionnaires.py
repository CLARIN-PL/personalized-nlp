import torch
import torch.nn as nn
import torch.nn.functional as F


class HuBiQuestionnaires(nn.Module):
    def __init__(
        self, output_dim, features_vector_length, text_embedding_dim=768, **kwargs
    ):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.fc1 = nn.Linear(text_embedding_dim + features_vector_length, output_dim)

    def forward(self, features):
        x = features["embeddings"]
        q_answers = features["q_answers"].float()

        x = x.view(-1, self.text_embedding_dim)
        x = self.fc1(torch.cat([x, q_answers], dim=1))
        return x
