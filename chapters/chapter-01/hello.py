import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)

x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor:", x)
print("Mean:", x.mean().item())