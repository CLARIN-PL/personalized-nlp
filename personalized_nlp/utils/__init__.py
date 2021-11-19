import torch
import random
import numpy as np


def seed_everything():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
