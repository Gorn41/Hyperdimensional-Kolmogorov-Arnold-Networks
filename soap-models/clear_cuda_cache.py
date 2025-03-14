import torch

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# Your operations go here

# Example dummy operation to test memory
x = torch.rand((1024, 1024), device='cuda')
y = torch.rand((1024, 1024), device='cuda')
z = torch.mm(x, y)

print(torch.cuda.memory_summary(device=None, abbreviated=False))

# Clean up GPU memory
del x, y, z
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
