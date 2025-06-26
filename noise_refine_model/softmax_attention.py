import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class SoftmaxAttention(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

       