import numpy as np
import tqdm
import random
import torch
import torchhd

class hdc_model:

    def __init__(self, hvsize, dims):
        self.hvsize = hvsize
        self.dims = dims
        self.codebook = {}

    def set(self, symbol):
        self.codebook[symbol] = torchhd.BSCTensor.random(1, self.hvsize, dtype=torch.long)

    def get(self, symbol):
        return self.codebook[symbol]
    
    def imagehv(self,image):
        res = torch.empty((self.dims ** 2, self.hvsize), dtype=torch.bool)
        for i in range(self.dims):
            for j in range(self.dims):
                value = image[i][j]
                if value not in self.codebook:
                    self.set(value)
                value_hv = self.get(value)
                for k in range((i * self.dims) + j):
                    value_hv = torchhd.permute(value_hv)
                res[i * self.dims + j] = value_hv
        return torchhd.multibundle(res)

    def decode_hv(self, image_hv):
        res = torch.empty((self.dims, self.dims))
        for i in range(self.dims):
            for j in range(self.dims):
                dists = {}
                for value in self.codebook.keys():
                    value_hv = self.get(value)
                    for k in range((i * self.dims) + j):
                        value_hv = torchhd.permute(value_hv)
                    dist = int(torchhd.hamming_similarity(image_hv, value_hv).item())
                    dists[dist] = value
                sorted_dists = sorted(dists.items(), reverse=True)
                largest = sorted_dists[0][1]
                print(largest.item())
                res[i][j] = largest.item()
        return res

    def save_codebook(self, path):
        with open(path, 'a', newline='') as file:
            file.write(self.codebook.items())
        return
    
    # save imagehv
    def save_imagehv(self, hv, label, path):
        with open(path, 'a', newline='') as file:
            file.write(hv + label + '\n')
        return
    
    def hdc_flatten_concat(self, tensor4D):
        batch_sz = tensor4D.shape[0]
        channels = tensor4D.shape[1]
        res = torch.empty((batch_sz, self.hvsize * channels))
        # iterate through examples in batch
        for i in range(batch_sz):
            tensor3D = tensor4D[i, :, :, :]
            # iterate through channels
            pre_concat = []
            for j in range(channels):
                tensor2D = tensor3D[j, :, :]
                hv = self.imagehv(tensor2D)
                pre_concat.append(hv.float())
            post = torch.cat(pre_concat)
            res[i] = post
        return res
    
        # try linear coords for 3d tenser
        # also try linear coords for 2d channel then concatenate
        # also try bundling/binding all three channels
        # send HDC model to device


# tensor1 = torch.tensor([1, 2, 3, 4, 5])
# tensor2 = torch.tensor([1, 5, 5, 5, 5])
# print(int(torchhd.hamming_similarity(tensor1, tensor1).item()))
# print(int(torchhd.hamming_similarity(tensor1, tensor2).item()))

# tensor4D = torch.rand(32, 2, 9, 9)
# model = hdc_model(10000, 9)
# print(model.HDC_flatten_concat(tensor4D).shape)


# # iterate through examples in batch
# for i in range(tensor4D.shape[0]):
#     tensor3D = tensor4D[i, :, :, :]
#     # iterate through channels
#     for j in range(tensor3D.shape[0]):
#         tensor2D = tensor3D[j, :, :]
#         hv = model.imagehv(tensor2D)
#         hv = torch.tensor(hv.float())
#         print(hv)
#         print(hv.shape)
#         print(type(hv))
#         # recovered = model.decode_hv(hv)
#         # print(tensor2D)
#         # print(recovered)
#         # print(recovered == tensor2D)
#         # print(recovered == torch.rand(9, 9))