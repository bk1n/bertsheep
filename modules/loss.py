import torch 
import numpy as np

class RMSE_loss(torch.nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        
    def forward(self, input, target):
        mse = (super().forward(input, target))
        rmse = torch.sqrt(mse)
        return rmse
    