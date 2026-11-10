import torch
print(torch.cuda.is_available(), torch.backends.mps.is_available() if hasattr(torch.backends,'mps') else False)