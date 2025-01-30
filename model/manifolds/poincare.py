import torch
from geoopt import PoincareBall

def artanh(x):
    return 0.5*torch.log((1+x)/(1-x))

class Poincare(PoincareBall):
    def __init__(self, k=1.0, learnable=False):
        super().__init__(k, learnable)
    