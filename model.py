import numpy as np
from numpy.random import choice, randint
import torch as th
from torch import nn
from torch.autograd import Function, Variable
from torch.utils.data import Dataset
from collections import defaultdict as ddict
from optimization import LorentzDistance
from optimization import LorentzInnerProduct
from optimization import eps

class LorentzEmbedding(nn.Module):
    def __init__(self, size, dim, dist=LorentzDistance):
        super(LorentzEmbedding, self).__init__() 
        self.dim = dim
        self.embeddings = nn.Embedding(
            size, dim,
            sparse=True,
            scale_grad_by_freq=False
        )
        self.dist = dist
        self.init_weights()
        self.lossfn = nn.CrossEntropyLoss

    def init_weights(self, scale=0.001):
        self.embeddings.state_dict()['weight'].uniform_(-scale, scale)
        #Place the vectors on the manifold
        components = self.embeddings.state_dict()['weight'][:,1:]
        self.embeddings.weight = nn.Parameter(th.cat([th.sqrt(th.sum(components * components, dim=1, keepdim=True) + 1), components], dim=1))

    def forward(self, inputs):
        e = self.embeddings(inputs)
        o = e[:,1:,:]
        s = e[:,0,:].unsqueeze(1).expand_as(o)
        dists = self.dist()(s, o).squeeze(-1)
        return -dists

    def loss(self, preds, targets, weight=None, size_average=True):
        lossfn = self.lossfn(size_average=size_average, weight=weight)
        return lossfn(preds, targets)

    def embedding(self):
        return list(self.embeddings.parameters())[0].data.cpu().numpy()
