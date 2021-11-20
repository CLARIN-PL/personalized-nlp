import torch
import torch.nn as nn
import torch.nn.functional as F


class WordBiasNet(nn.Module):
    def __init__(
        self,
        output_dim,
        text_embedding_dim,
        word_num,
        dp=0.0,
        hidden_dim=100,
        **kwargs
    ):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.hidden_dim = hidden_dim

        self.word_biases = torch.nn.Embedding(
            num_embeddings=word_num, embedding_dim=output_dim, padding_idx=0
        )

        self.word_biases.weight.data.uniform_(-0.001, 0.001)

        self.dp = nn.Dropout(p=dp)

        self.fc1 = nn.Linear(self.text_embedding_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, output_dim)

        self.softplus = nn.Softplus()

    def forward(self, features):
        x = features["embeddings"]
        tokens = features["tokens_sorted"].long()

        x = x.view(-1, self.text_embedding_dim)
        x = self.fc1(x)
        x = self.softplus(x)

        word_biases = self.word_biases(tokens)

        mask = tokens != 0
        word_biases = (word_biases * mask[:, :, None]).sum(dim=1)

        x = self.fc2(x) + word_biases

        return x
