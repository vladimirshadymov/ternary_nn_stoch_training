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
    def __init__(self, min=-1, max=1, discrt_lvls=2):
        super(Discretization, self).__init__()
        self.min = min
        self.max = max
        self.discrt_lvls = discrt_lvls

    def forward(self, input):
        return 0.5*(DiscreteHardtanhFunction.apply(input, self.discrt_lvls)*(self.max - self.min) + self.min + self.max)

class DiscretizedLinear(nn.Linear):

    def __init__(self, min_weight=-1, max_weight=1, discrt_lvls=2, *kargs, **kwargs):
        super(DiscretizedLinear, self).__init__(*kargs, **kwargs)
        self.discretization = Discretization(min=min_weight, max=max_weight, discrt_lvls=discrt_lvls)
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.discrt_lvls = discrt_lvls

    def forward(self, input):
        self.weight.data = F.hardtanh_(self.weight.data)
        return F.linear(input, self.discretization(self.weight), bias=self.bias)
