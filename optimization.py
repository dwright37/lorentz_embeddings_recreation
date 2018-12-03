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
        #if th.isnan(g).any():
        #    print("arcosh")
        #    print(g)
        #    exit()
        z = th.clamp(self.z, min=eps)
        z = g / z 
        #if th.isnan(z).any():
        #    print("arcosh_back")
        #    print(z)
        #    exit()
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
def lorentz_retraction(gradient):
    g = th.eye(gradient.shape[-1])
    g[0,0] = -1.0
    return th.mm(gradient, g)

def proj(theta, ht):
    val = ht + LorentzInnerProduct()(theta, ht) * theta
    #lip = LorentzInnerProduct()(theta, val)
    #if (lip > 0.1).any() or (lip < -0.1).any():
    #    print("NOT IN TANGENT SPACE")
    #    print(lip)
    #    exit()
    return val

def expmap(theta, v):
    #print("expmap")
    normv = th.clamp(th.sqrt(LorentzInnerProduct()(v,v)), min=eps)
    val = th.cosh(normv) * theta + th.sinh(normv) * (v / normv)
    #if (val[:,0] < 0).any():
    #    print("invalid embedding")
    #    exit()
    return val

class RiemannianSGD(Optimizer):
    def __init__(self, params, lr=0.001):
        defaults = dict(lr=lr)
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
                    #print(grad_data)
                    grad_indices = gradient._indices()[0].squeeze()
                    #grad_indices = gradient._indices()

                    param_data = p.data[grad_indices]
                    #print(param_data)
                    #print(param_data.shape)
                    #if th.isnan(grad_data).any():
                    #    print("euclidean gradient")
                    #    print(grad_data)
                    #    print(grad_indices)
                    #    exit()
                    
                    newpoints = expmap(param_data, -lr*proj(param_data, lorentz_retraction(grad_data)))
                    components = newpoints[:,1:]
                    #print(newpoints[:,0].shape)
                    dim0 = th.sqrt(th.sum(components * components, dim=1, keepdim=True) + 1)
                    newpoints2 = th.cat([dim0, components], dim=1)
                    #print(newpoints.shape)
                    #print(gradient.size())
                    #newgrad = gradient.new(grad_indices.unsqueeze(0), newpoints, gradient.size())
                    #print(dim0.shape)
                    #print(newpoints)
                    p.data[grad_indices] = newpoints2
                    #print(p.data[grad_indices][1,0] - p.data[grad_indices][1,0])
                    #print(p.data[grad_indices][1,0] - newpoints[1,0])
                    #print(p.data[grad_indices][1,0] - dim0[1])
                    #print(newpoints[1, 0])
                    #print(dim0[1])
                    #print(p.data[grad_indices][1,0])
                    #old = p.data[grad_indices][:,0]
                    #p.data[grad_indices][:,0] = th.sqrt(th.sum(components * components, dim=1) + 1)
                    #print(p.data[grad_indices][:,0] - th.sqrt(th.sum(components * components, dim=1) + 1))
                    #lip = LorentzInnerProduct()(p.data[grad_indices], p.data[grad_indices]).squeeze()
                    #if (lip > -0.5).any() or (lip < -1.5).any():
                    #    print("wtf")
                    #    print(p.data[grad_indices][lip > -0.9])
                    #    #print(old[lip > -0.9])
                    #    print(th.sqrt(th.sum(components * components, dim=1) + 1)[lip > -0.9])
                    #    print(p.data[grad_indices][lip < -1.1])
                    #    #print(old[lip < -1.1])
                    #    print(th.sqrt(th.sum(components * components, dim=1) + 1)[lip < -1.1])
                        
                    #    exit()
            
                else:
                    param_data = p.data
                    newpoints = expmap(param_data, -lr*proj(param_data, lorentz_retraction(gradient)))
                    components = newpoints[:,1:]
                    dim0 = th.sqrt(th.sum(components * components, dim=1, keepdim=True) + 1)
                    newpoints2 = th.cat([dim0, components], dim=1)
                    p.data = newpoints2                
                    #lip = LorentzInnerProduct()(p.data, p.data).squeeze()
                    #if (lip > -0.5).any() or (lip < -1.5).any():
                    #    print("wtf")
                    #    exit()

        return loss

