from collections import defaultdict
import torch as th
import numpy as np
from torch.utils.data import Dataset
from numpy.random import choice, randint
from itertools import count

class DatasetReader(Dataset):
    def __init__(self, dataset_file, nnegs, unigram_size=1e8):
        self.nnegs = nnegs
        self.max_tries = nnegs * 5
        self.read_data(dataset_file, unigram_size)
        
    def read_data(self, dataset_file, unigram_size=1e8):
        entities = defaultdict(count().__next__)
        self.weights = defaultdict(lambda: defaultdict(lambda: 0))
        self.samples = []
        counts = defaultdict(lambda: 1)
        with open(dataset_file) as f:
            for l in f:
                fields = l.strip().split('\t')
                self.weights[entities[fields[0]]][entities[fields[1]]] = int(fields[2])
                self.samples.append((entities[fields[0]], entities[fields[1]], int(fields[2])))
                counts[entities[fields[0]]] += 1
        self.entities = [None for i in range(len(entities))]
        for e in entities:
            self.entities[entities[e]] = e
        self._counts = np.ones(len(self.entities), dtype=np.float)
        for e in counts:
            self._counts[e] = counts[e]

        self.unigram_table = choice(
            len(self.entities),
            size=int(unigram_size),
            p = (self._counts / self._counts.sum())
        )
        
            
    def __len__(self):
        return len(self.samples)
            
    def __getitem__(self, i):
        left, right, weight = self.samples[i]
        negs = set()
        nnegs = self.nnegs
        if self.burnin:
            nnegs *= 0.1
        tries = 0
        while tries < self.max_tries and len(negs) < nnegs:
            if self.burnin:
                n = randint(0, len(self.unigram_table))
                n = int(self.unigram_table[n])
            else:
                n = randint(0, len(self.entities))
            
            if self.weights[left][n] < weight:
                negs.add(n)
                
            tries += 1
        if len(negs) == 0:
            negs = [t]
        ex = [left, right] + list(negs)
        while len(ex) < nnegs + 2:
            ex.append(ex[randint(2, len(ex))])
            
        return np.asarray(ex), np.zeros(1).astype(np.long)
    
    @classmethod
    def collate(cls, batch):
        return zip(*batch)
