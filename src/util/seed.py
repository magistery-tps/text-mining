import numpy  as np
import torch
import random


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)