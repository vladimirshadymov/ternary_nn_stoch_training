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
        output[output < 0] = -1.

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

    def __init__(self, scale_factor):
        super(Binarization, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        return BinarizeFunction.apply(input).mul(self.scale_factor)

class BinarizedLinear(nn.Linear):

    def __init__(self, scale_factor=1e-2, *kargs, **kwargs):
        super(BinarizedLinear, self).__init__(*kargs, **kwargs)
        self.binarization = Binarization(scale_factor=scale_factor)
        # self.weight.data.uniform_(min_weight, max_weight)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-max_weight, -min_weight)
        self.scale_factor = scale_factor
        self.noise_on = False
        self.weight_noise_percent = None
        self.input_noise_percent = None
        self.weight_noise = None
        self.input_noise = None

    def forward(self, input):
        if self.noise_on:
            device_num = self.weight.get_device()
            device = torch.device("cuda:%d" % device_num)
            self.input_noise = torch.normal(mean=0.0, std=torch.ones_like(input).mul(self.input_noise_percent)).add(1.).to(device)
            if not self.weight_noise is None:
                self.weight_noise = self.weight_noise.to(device)
            # print(self.input_noise.abs().mean().item())
            # print(self.weight_noise.abs().mean().item())
        self.weight.data = nn.functional.hardtanh_(self.weight.data, min_val=-1., max_val=1.)

        if self.noise_on:
            out = nn.functional.linear(input.mul(self.input_noise), self.binarization(self.weight)+self.weight_noise, bias=self.bias)
        else:
            out = nn.functional.linear(input, self.binarization(self.weight), bias=self.bias)  # linear layer with binarized weights
        return out

    def quantize_accumulative_weigths(self):
        self.weight.data = self.binarization(self.weight.data)
        return
    
    def set_weight_noise_val(self, percent=0.2):
        self.noise_on = True
        self.weight_noise_percent = percent
        self.weight_noise = torch.normal(mean=0.0, std=torch.ones_like(self.weight.data)*self.weight_noise_percent*self.scale_factor)
        return

    def set_input_noise_val(self, percent=0.2):
        self.noise_on = True
        self.input_noise_percent = percent
        # self.input_noise = torch.normal(mean=0.0, std=self.ones_input*(1+self.input_noise_percent))
        return
    
    def set_noise(self, noise_on=True):
        self.noise_on = noise_on
        return

class BinarizedConv2d(nn.Conv2d):

    def __init__(self, scale_factor, *kargs, **kwargs):
        super(BinarizedConv2d, self).__init__(*kargs, **kwargs)
        self.scale_factor = scale_factor
        self.binarization = Binarization(scale_factor=scale_factor)
        self.noise_on = False
        self.weight_noise_percent = 0.2
        self.input_noise_percent = 0.2
        self.weight_noise = None
        self.input_noise = None
        
    def forward(self, input):
        if self.noise_on:
            device_num = self.weight.get_device()
            device = torch.device("cuda:%d" % device_num)
            self.input_noise = torch.normal(mean=0.0, std=torch.ones_like(input).mul(self.input_noise_percent)).add(1.).to(device)
            if not self.weight_noise is None:
                self.weight_noise = self.weight_noise.to(device)
        self.weight.data = nn.functional.hardtanh_(self.weight.data, min_val=-1., max_val=1.)

        if self.noise_on:
            return nn.functional.conv2d(input.mul(self.input_noise), self.binarization(self.weight)+self.weight_noise, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
        else:
            return nn.functional.conv2d(input, self.binarization(self.weight), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)

    def quantize_accumulative_weigths(self):
        self.weight.data = self.binarization(self.weight.data)
        return
    
    def set_weight_noise_val(self, percent=0.2):
        self.noise_on = True
        self.weight_noise_percent = percent
        self.weight_noise = torch.normal(mean=0.0, std=torch.ones_like(self.weight.data)*self.weight_noise_percent*self.scale_factor)
        return

    def set_input_noise_val(self, percent=0.2):
        self.noise_on = True
        self.input_noise_percent = percent
        # self.input_noise = torch.normal(mean=0.0, std=self.ones_input*(1+self.input_noise_percent))
        return
    
    def set_noise(self, noise_on=True):
        self.noise_on = noise_on
        return

class TernarizeFunction(Function):

    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input)
        ctx.threshold = threshold

        output = input.clone()
        output[output.abs() <= threshold] = 0
        output[output > threshold] = 1.
        output[output < -threshold] = -1.

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = input.clone()
        grad_input[torch.abs(input) <= 1.] = 1.
        grad_input[torch.abs(input) > 1.] = 0.
        grad_input = grad_input * grad_output

        return grad_input, None

class Ternarization(nn.Module):

    def __init__(self, scale_factor, threshold):
        super(Ternarization, self).__init__()
        self.scale_factor = scale_factor
        self.threshold = threshold

    def forward(self, input):
        return TernarizeFunction.apply(input, self.threshold).mul(self.scale_factor)

class TernarizedLinear(nn.Linear):

    def __init__(self, scale_factor, threshold, *kargs, **kwargs):
        super(TernarizedLinear, self).__init__(*kargs, **kwargs)
        self.ternarization = Ternarization(scale_factor=scale_factor, threshold=threshold)
        self.scale_factor = scale_factor
        self.threshold = threshold
        self.noise_on = False
        self.weight_noise_percent = 0.2
        self.input_noise_percent = 0.2
        self.weight_noise = None
        self.input_noise = None

    def forward(self, input):
        if self.noise_on:
            device_num = self.weight.get_device()
            device = torch.device("cuda:%d" % device_num)
            self.input_noise = torch.normal(mean=0.0, std=torch.ones_like(input).mul(self.input_noise_percent)).add(1.).to(device)
            if not self.weight_noise is None:
                self.weight_noise = self.weight_noise.to(device)
        self.weight.data = nn.functional.hardtanh_(self.weight.data, min_val=-1., max_val=1.)

        if self.noise_on:
            out = nn.functional.linear(input.mul(self.input_noise), self.ternarization(self.weight)+self.weight_noise, bias=self.bias)
        else:
            out = nn.functional.linear(input, self.ternarization(self.weight), bias=self.bias)  # linear layer with binarized weights
        return out

    def quantize_accumulative_weigths(self):
        self.weight.data = self.ternarization(self.weight.data)
        return
    
    def set_weight_noise_val(self, percent=0.2):
        self.noise_on = True
        self.weight_noise_percent = percent
        self.weight_noise = torch.normal(mean=0.0, std=torch.ones_like(self.weight.data)*self.weight_noise_percent*self.scale_factor)
        return

    def set_input_noise_val(self, percent=0.2):
        self.noise_on = True
        self.input_noise_percent = percent
        # self.input_noise = torch.normal(mean=0.0, std=self.ones_input*(1+self.input_noise_percent))
        return
    
    def set_noise(self, noise_on=True):
        self.noise_on = noise_on
        return

    def add_bit_error(self, bit_error_rate = 0):
        probs = torch.ones_like(self.weight.data).mul_(1 - bit_error_rate) # switching probabilities
        switching_tensor = torch.bernoulli(probs).mul(2.).add(-1.)
        self.weight.data.mul_(switching_tensor)
        return

    def set_random_binary_weights(self, prob = 0.5):
        self.weight.data = torch.bernoulli(torch.ones_like(self.weight.data)*prob).mul_(2.).add_(-1.)
        return

class TernarizedConv2d(nn.Conv2d):

    def __init__(self, scale_factor, threshold, *kargs, **kwargs):
        super(TernarizedConv2d, self).__init__(*kargs, **kwargs)
        self.scale_factor = scale_factor
        self.threshold = threshold
        self.ternarization = Ternarization(scale_factor=scale_factor, threshold=threshold)
        self.noise_on = False
        self.weight_noise_percent = None
        self.input_noise_percent = None
        self.weight_noise = None
        self.input_noise = None

    def forward(self, input):
        if self.noise_on:
            device_num = self.weight.get_device()
            device = torch.device("cuda:%d" % device_num)
            self.input_noise = torch.normal(mean=0.0, std=torch.ones_like(input).mul(self.input_noise_percent)).add(1.).to(device)
            if not self.weight_noise is None:
                self.weight_noise = self.weight_noise.to(device)     
        self.weight.data = nn.functional.hardtanh_(self.weight.data, min_val=-1., max_val=1.)

        if self.noise_on:
            return nn.functional.conv2d(input.mul(self.input_noise), self.ternarization(self.weight)+self.weight_noise, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
        else:
            return nn.functional.conv2d(input, self.ternarization(self.weight), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)

    def quantize_accumulative_weigths(self):
        self.weight.data = self.binarization(self.weight.data)
        return
    
    def set_weight_noise_val(self, percent=0.2):
        self.noise_on = True
        self.weight_noise_percent = percent
        self.weight_noise = torch.normal(mean=0.0, std=torch.ones_like(self.weight.data)*self.weight_noise_percent*self.scale_factor)
        return

    def set_input_noise_val(self, percent=0.2):
        self.noise_on = True
        self.input_noise_percent = percent
        # self.input_noise = torch.normal(mean=0.0, std=self.ones_input*(1+self.input_noise_percent))
        return
    
    def set_noise(self, noise_on=True):
        self.noise_on = noise_on
        return

class StochasticTernaryLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight_p, weight_n, bias=None, min_val=-1, max_val=+1):
        ctx.save_for_backward(input, weight_p, weight_n, bias)
        ctx.min_val = min_val
        ctx.max_val = max_val
        
        weight_p.data = torch.where(weight_p.data>=0, torch.zeros_like(weight_p).add(1.0), torch.zeros_like(weight_p).add(-1.0))
        weight_n.data = torch.where(weight_n.data>=0, torch.zeros_like(weight_n).add(1.0), torch.zeros_like(weight_n).add(-1.0))

        output = input.mm(weight_p.add(-weight_n).mul(max_val - min_val).add(min_val + max_val).mul(0.25).t())
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
        time_lr = 150

        weight_p.data = torch.where(weight_p.data >=0, torch.zeros_like(weight_p).add(1.0), torch.zeros_like(weight_p).add(-1.0))
        weight_n.data = torch.where(weight_n.data >=0, torch.zeros_like(weight_n).add(1.0), torch.zeros_like(weight_n).add(-1.0))

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_p.add(-weight_n).mul(max_val - min_val).add(min_val + max_val).mul(0.25))

        if ctx.needs_input_grad[1]:
            grad_weight_p = grad_output.t().mm(input)
            
            tmp = grad_output.abs().mul(time_lr*switching_time_add).add(switching_time_critical).div(-T*0.5).t().mm(torch.ones_like(input))
            tmp = torch.where(weight_p.data>=0, tmp.mul(high_to_low_cond_current_add1).mul(input.abs()).add(high_to_low_cond_current_add0-high_to_low_cond_current_critical), tmp.mul(low_to_high_cond_current_add1).mul(input.abs()).add(low_to_high_cond_current_add0-low_to_high_cond_current_critical))
            tmp.exp_().mul_(-4*delta) 
            tmp2 = torch.zeros_like(weight_p.data).add(input.abs())
            tmp2 = torch.where(weight_p.data>=0, tmp2.abs().mul(high_to_low_cond_current_add1).add(high_to_low_cond_current_add0).div(high_to_low_cond_current_critical), tmp2.abs().mul(low_to_high_cond_current_add1).add(low_to_high_cond_current_add0).div(low_to_high_cond_current_critical))
            tmp2 = torch.pow(tmp2.mul(2).div(tmp2.add(-1)), -2/(tmp2.add(1)))
            tmp.mul_(tmp2).exp_()
            # print(f'probability p: {tmp.max().item(), tmp.mean().item()}')
            del tmp2
            tmp = torch.bernoulli(tmp)

            grad_weight_p.data.sign_().mul_(1e7)
            grad_weight_p.data.mul_(tmp)

            del tmp
        
        if ctx.needs_input_grad[2]:
            grad_weight_n = grad_output.t().mm(-input)
            
            tmp = grad_output.abs().mul(time_lr*switching_time_add).add(switching_time_critical).div(-T*0.5).t().mm(torch.ones_like(input))
            tmp = torch.where(weight_n.data>=0, tmp.mul(high_to_low_cond_current_add1).mul(input.abs()).add(high_to_low_cond_current_add0-high_to_low_cond_current_critical), tmp.mul(low_to_high_cond_current_add1).mul(input.abs()).add(low_to_high_cond_current_add0-low_to_high_cond_current_critical))
            tmp.exp_().mul_(-4*delta) 
            tmp2 = torch.zeros_like(weight_n.data).add(input.abs())
            tmp2 = torch.where(weight_n.data>=0, tmp2.abs().mul(high_to_low_cond_current_add1).add(high_to_low_cond_current_add0).div(high_to_low_cond_current_critical), tmp2.abs().mul(low_to_high_cond_current_add1).add(low_to_high_cond_current_add0).div(low_to_high_cond_current_critical))
            tmp2 = torch.pow(tmp2.mul(2).div(tmp2.add(-1)), -2/(tmp2.add(1)))
            tmp.mul_(tmp2).exp_()
            # print(f'probability n: {tmp.max().item(), tmp.mean().item()}')
            del tmp2
            tmp = torch.bernoulli(tmp)

            grad_weight_n.data.sign_().mul_(1e7)
            grad_weight_n.data.mul_(tmp)

            del tmp            

        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0)

        # print(f'delta: {grad_output.abs().mean()}')
        # print(f'input: {input.abs().mean()}')
        # print(f'grad_weight_p: {grad_weight_p.abs().max()}')
        # print(f'grad_weight_n: {grad_weight_n.abs().max()}')


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
