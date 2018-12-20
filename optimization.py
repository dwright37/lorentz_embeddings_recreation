import numpy as np
import torch as th
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer

#######################################
# Arcosh
#######################################
eps = 1e-8
class Arcosh(Function):
    def __init__(self, eps=eps):
        super(Arcosh, self).__init__()
        self.eps = eps 

    def forward(self, x): 
        self.z = th.sqrt(th.clamp(x * x - 1, min=eps))
        return th.log(th.clamp(x + self.z, min=eps))

    def backward(self, g): 
        z = th.clamp(self.z, min=eps)
        z = g / z 
        return z

########################################
# Lorentz inner product
########################################
class LorentzInnerProduct(nn.Module):
    def __init__(self):
        super(LorentzInnerProduct, self).__init__()
        
    def forward(self, x, y):
        z = x*y
        findex = len(z.size()) - 1
        rng = z.size()[-1] - 1
        val = th.sum(z.narrow(findex, 1, rng), dim=-1, keepdim=True) - z.narrow(findex, 0, 1)
        return val
    
########################################
# Lorentz distance
########################################
class LorentzDistance(nn.Module):
    def __init__(self):
        super(LorentzDistance, self).__init__()
        self.lip = LorentzInnerProduct()
        self.arcosh = Arcosh()
    def forward(self, x, y):
        lip = th.clamp(-self.lip(x,y), min=1.0 + eps)
        d =  self.arcosh(lip)
        return d
    
########################################
# Riemannian SGD
########################################
def lorentz_retraction(gradient, device='cpu'):
    g = th.eye(gradient.shape[-1]).to(device)
    g[0,0] = -1.0
    return th.mm(gradient, g)

def proj(theta, ht):
    val = ht + LorentzInnerProduct()(theta, ht) * theta
    return val

def expmap(theta, v):
    normv = th.clamp(th.sqrt(LorentzInnerProduct()(v,v)), min=eps)
    val = th.cosh(normv) * theta + th.sinh(normv) * (v / normv)
    return val

class RiemannianSGD(Optimizer):
    def __init__(self, params, lr=0.001, device='cpu'):
        defaults = dict(lr=lr)
        self.device = device
        super(RiemannianSGD, self).__init__(params, defaults)
    
    def step(self, lr=None):
        loss = None
        
        for group in self.param_groups:
            if lr is None:
                lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                gradient = p.grad.data
                if gradient.is_sparse:
                    gradient = gradient.coalesce()
                    grad_data = gradient._values()
                    grad_indices = gradient._indices()[0].squeeze()

                    param_data = p.data[grad_indices]
                    
                    newpoints = expmap(param_data, -lr*proj(param_data, lorentz_retraction(grad_data, device=self.device)))
                    components = newpoints[:,1:]
                    dim0 = th.sqrt(th.sum(components * components, dim=1, keepdim=True) + 1)
                    newpoints2 = th.cat([dim0, components], dim=1)
                    p.data[grad_indices] = newpoints2
                else:
                    param_data = p.data
                    newpoints = expmap(param_data, -lr*proj(param_data, lorentz_retraction(gradient, device=self.device)))
                    components = newpoints[:,1:]
                    dim0 = th.sqrt(th.sum(components * components, dim=1, keepdim=True) + 1)
                    newpoints2 = th.cat([dim0, components], dim=1)
                    p.data = newpoints2                

        return loss

