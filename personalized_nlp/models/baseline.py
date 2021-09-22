import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, output_dim=2, text_embedding_dim=768, **kwargs):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.fc1 = nn.Linear(text_embedding_dim, output_dim) 

    def forward(self, features):
        x = features['embeddings']
        x = x.view(-1, self.text_embedding_dim)
        x = self.fc1(x)

        return x
