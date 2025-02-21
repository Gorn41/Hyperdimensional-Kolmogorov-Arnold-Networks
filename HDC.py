import numpy as np
import tqdm
import random
import torch
import torchhd
import CSV

class HDC:

    def __init__(self, hvsize, dims):
        self.hvsize = hvsize
        self.dims = dims
        self.codebook = {}

    def set(self, symbol):
        self.codebook[symbol] = torchhd.BSCTensor.random(1, self.hvsize, dtype=torch.long)

    def get(self, symbol):
        return self.codebook[symbol]
    
    def imagehv(self,image):
        res = torch.empty((self.hvsize ** 2,), dtype=torch.bool)
        for i in range(self.dims):
            for j in range(self.dims):
                value = image.getpixel((i, j))
                if value not in self.codebook:
                    self.set(value)
                value_hv = self.get(value)
                for k in range((i * self.dims) + j):
                    value_hv = torchhd.permute(value_hv)
                res[i * self.dims + j]
        return torchhd.multibundle(res)

    def save_codebook(self, path):
        with open(path, 'a', newline='') as file:
            file.write(self.codebook.items())
        return
    
    # save imagehv
    def save_imagehv(self, hv, label, path):
        with open(path, 'a', newline='') as file:
            file.write(hv + label + '\n')
        return
    


