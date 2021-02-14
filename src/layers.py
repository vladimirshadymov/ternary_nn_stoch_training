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

        return F.hardtanh(input).add(1.).mul((discrt_lvls-1)/2).round().div((discrt_lvls-1)/2).add(-1.)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        grad_input = input.clone()
        grad_input[torch.abs(input) <= 1 ] = 1.
        grad_input[torch.abs(input) > 1 ] = 0.
        grad_input = grad_input * grad_output

        return grad_input, None

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
        # print(self.discretization(self.weight).unique())
        return F.linear(input, self.discretization(self.weight), bias=self.bias)

class DiscretizedConv2d(nn.Conv2d):

    def __init__(self, min_weight=-1, max_weight=1, discrt_lvls=2, *kargs, **kwargs):
        super(DiscretizedConv2d, self).__init__(*kargs, **kwargs)
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.discrt_lvls = discrt_lvls
        self.discretization = Discretization(min=min_weight, max=max_weight, discrt_lvls=discrt_lvls)

    def forward(self, input):
        self.weight.data = nn.functional.hardtanh_(self.weight.data)
        return F.conv2d(input, self.discretization(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)

class BinarizeFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        output = input.clone()
        output[output >= 0] = 1.
        output[output < 0] = -1

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = input.clone()
        grad_input[torch.abs(input) <= 1.] = 1.
        grad_input[torch.abs(input) > 1.] = 0.
        grad_input = grad_input * grad_output

        return grad_input

class Binarization(nn.Module):

    def __init__(self, min=-1, max=1):
        super(Binarization, self).__init__()
        self.min = min
        self.max = max

    def forward(self, input):
        return 0.5*(BinarizeFunction.apply(input)*(self.max - self.min) + self.min + self.max)

class StochasticTernaryLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight_p, weight_n, bias=None, min_val=-1, max_val=+1):
        ctx.save_for_backward(input, weight_p, weight_n, bias)
        ctx.min_val = min_val
        ctx.max_val = max_val
        
        weight_p.data = torch.where(weight_p.data>=0, torch.zeros_like(weight_p).add(1.0), torch.zeros_like(weight_p).add(-1.0))
        weight_n.data = torch.where(weight_n.data>=0, torch.zeros_like(weight_n).add(1.0), torch.zeros_like(weight_n).add(-1.0))

        output = input.mm(weight_p.add(-weight_n).mul(0.5).t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight_p, weight_n, bias, = ctx.saved_tensors
        grad_input = grad_weight_p = grad_weight_n = grad_bias = None
        min_val = ctx.min_val
        max_val = ctx.max_val   

        current_add0 = 60
        current_add1 = 30

        low_to_high_cond_current_critical = 64.5 # mkA critical current to switch from low to high cell's conductivity. From Parallel to Anti-Parallel State
        low_to_high_cond_current_add0 = 140  
        low_to_high_cond_current_add1 = 60 # mkA additional current to switch from low to high cell's conductivity. From Parallel to Anti-Parallel State

        high_to_low_cond_current_critical = 21.2 # mkA critical current to switch from high to low cell's conductivity. From Anti-Parallel to Parallel State
        high_to_low_cond_current_add0 = 60
        high_to_low_cond_current_add1 = 30 # mkA additional current to switch from high to low cell's conductivity. From Anti-Parallel to Parallel State

        switching_time_critical = 1.5 # ns
        switching_time_add = 1.0 # ns
        
        delta = 40.88#-15
        T = 57.09#+4
        time_lr = 0.7

        weight_p.data = torch.where(weight_p.data >=0, torch.zeros_like(weight_p).add(1.0), torch.zeros_like(weight_p).add(-1.0))
        weight_n.data = torch.where(weight_n.data >=0, torch.zeros_like(weight_n).add(1.0), torch.zeros_like(weight_n).add(-1.0))

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)

        if ctx.needs_input_grad[1]:
            grad_weight_p = grad_output.t().mm(input)
            
            tmp = grad_output.abs().mul(time_lr*switching_time_add).add(switching_time_critical).div(-T*0.5).t().mm(torch.ones_like(input))
            tmp = torch.where(weight_p.data>=0, tmp.mul(high_to_low_cond_current_add1).mul(input.abs()).add(high_to_low_cond_current_add0-high_to_low_cond_current_critical), tmp.mul(low_to_high_cond_current_add1).mul(input.abs()).add(low_to_high_cond_current_add0-low_to_high_cond_current_critical))
            tmp.exp_().mul_(-4*delta) 
            tmp2 = torch.zeros_like(weight_p.data).add(input.abs())
            tmp2 = torch.where(weight_p.data>=0, tmp2.abs().mul(high_to_low_cond_current_add1).add(high_to_low_cond_current_add0).div(high_to_low_cond_current_critical), tmp2.abs().mul(low_to_high_cond_current_add1).add(low_to_high_cond_current_add0).div(low_to_high_cond_current_critical))
            tmp2 = torch.pow(tmp2.mul(2).div(tmp2.add(-1)), -2/(tmp2.add(1)))
            tmp.mul_(tmp2).exp_()
            del tmp2
            tmp = torch.bernoulli(tmp)

            grad_weight_p.data.sign_().mul_(1e7)
            grad_weight_p.data.mul_(tmp)

            del tmp
        
        if ctx.needs_input_grad[2]:
            grad_weight_n = grad_output.t().mm(input)
            
            tmp = grad_output.abs().mul(time_lr*switching_time_add).add(switching_time_critical).div(-T*0.5).t().mm(torch.ones_like(input))
            tmp = torch.where(weight_n.data>=0, tmp.mul(high_to_low_cond_current_add1).mul(input.abs()).add(high_to_low_cond_current_add0-high_to_low_cond_current_critical), tmp.mul(low_to_high_cond_current_add1).mul(input.abs()).add(low_to_high_cond_current_add0-low_to_high_cond_current_critical))
            tmp.exp_().mul_(-4*delta) 
            tmp2 = torch.zeros_like(weight_n.data).add(input.abs())
            tmp2 = torch.where(weight_n.data>=0, tmp2.abs().mul(high_to_low_cond_current_add1).add(high_to_low_cond_current_add0).div(high_to_low_cond_current_critical), tmp2.abs().mul(low_to_high_cond_current_add1).add(low_to_high_cond_current_add0).div(low_to_high_cond_current_critical))
            tmp2 = torch.pow(tmp2.mul(2).div(tmp2.add(-1)), -2/(tmp2.add(1)))
            tmp.mul_(tmp2).exp_()
            del tmp2
            tmp = torch.bernoulli(tmp)

            grad_weight_n.data.sign_().mul_(1e7)
            grad_weight_n.data.mul_(tmp)

            del tmp            

        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight_p, grad_weight_n, grad_bias, None, None

class StochasticTernaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, min_val=-1, max_val=+1):
        super(StochasticTernaryLinear, self).__init__()
        self.func = StochasticTernaryLinearFunction.apply
        self.min_val = min_val
        self.max_val = max_val
        self.in_features = in_features
        self.out_features = out_features
        self.weight_p = Parameter(torch.Tensor(out_features, in_features))
        self.weight_n = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.bias = None

        self.weight_p.data.uniform_(-0.1, 0.1)
        self.weight_n.data.uniform_(-0.1, 0.1)

        if self.bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        return self.func(input, self.weight_p, self.weight_n, self.bias, self.min_val, self.max_val)
