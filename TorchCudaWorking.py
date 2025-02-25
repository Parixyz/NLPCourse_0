import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)

result = x @ y  # Matrix multiplication on GPU
print(result)
