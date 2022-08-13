from typing import Optional

import torch
import numpy as np
import pytorch_lightning as pl

from torch import nn


class FlowModel(pl.LightningModule):
    
    def __init__(
        self,
        flow_model: nn.Module,
        class_dims=None, 
        lr: float=1e-4, 
        class_names=None,
    ):
        super(FlowModel, self).__init__()
        self.flow_model = flow_model
        self.lr = lr

        self.class_dims = class_dims
        self.class_names = class_names
        
    def forward(self, x, y):
        z = self.flow_model.log_prob(batch=x, y=y)
        return z
    
    def sample(self, x, y, n, bs):
        return self.flow_model.sample(batch=x, y=y, num_samples=n, batch_size=bs)

    def step(self, x, y):
        loss = - self.flow_model.log_prob(batch=x, y=y).mean()
        #raise Exception(f'Loss: {loss.shape} x: {x["embeddings"].shape} y: {y.shape}')
        return loss
        

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch

        #output = self.forward(x)
        loss = self.step(x=x, y=y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        #output = self.forward(x)
        loss = self.step(x=x, y=y)

        self.log("valid_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        loss = self.step(x=x, y=y)
        #z = self.sample(x, y, y.shape[0], y.shape[0])
        #raise Exception(z.shape)

        self.log("test_loss", loss, prog_bar=True)

        #return {'loss': loss.cpu().numpy(), 'text_ids': x['text_ids'].cpu().numpy(), 'annotator_ids': x['annotator_ids'].cpu().numpy(), 'y': y.cpu().numpy(), 'pz': z.cpu().numpy()}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer