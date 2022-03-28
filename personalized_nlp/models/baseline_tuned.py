import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class NetTuned(nn.Module):

    def __init__(self, output_dim=2, text_embedding_dim=768, **kwargs):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.fc1 = nn.Linear(text_embedding_dim, output_dim)
        self.is_frozen = False

    def forward(self, features):
        x = features['embeddings']
        x = x.view(-1, self.text_embedding_dim)
        x = self.fc1(x)
        return x

    def freeze(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.is_frozen = True

    def unfreeze(self) -> None:
        if self.is_frozen:
            for name, param in self.named_parameters():
                param.requires_grad = True
        self.is_frozen = False

    def on_epoch_start(self):
        if self.current_epoch < self.hparams.frozen_epochs:
            self.freeze()

        if self.current_epoch >= self.hparams.frozen_epochs:
            self.unfreeze()
