# Import PyTorch
import torch  # import main library
from torch.autograd import Function  # import Function to create custom activations
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import math
import numpy as np

class DiscreteHardtanhFunction(Function):

    @staticmethod
    def forward(ctx, input, discrt_lvls=2):
        ctx.save_for_backward(input)  # save input for backward pass

        return F.hardtanh(input).add(1.).mul(discrt_lvls/2).round().div(discrt_lvls/2).add(-1.)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        grad_input = input.clone()
        grad_input[torch.abs(input) <= 1 ] = 1.
        grad_input[torch.abs(input) > 1 ] = 0.
        grad_input = grad_input * grad_output

        return grad_input

class Discretization(nn.Module):
    def __init__(self, min=-1, max=1):
        super(Binarization, self).__init__()
        self.min = min
        self.max = max

    def forward(self, input):
        return 0.5*(DiscreteHardtanhFunction.apply(input)*(self.max - self.min) + self.min + self.max)
