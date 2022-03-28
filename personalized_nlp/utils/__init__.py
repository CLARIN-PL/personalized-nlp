import torch
import random
import numpy as np


def seed_everything(seed=22):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
