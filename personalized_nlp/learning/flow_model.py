import torch
import numpy as np
import pytorch_lightning as pl

from torch import nn


class FlowModel(pl.LightningModule):
    
    def __init__(
        self,
        model, 
        flow,
        class_dims, 
        lr: float, 
        class_names=None,
    ):
        super(FlowModel, self).__init__()
        

