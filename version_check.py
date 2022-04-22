import torch
print(torch.__version__)
tens = torch.tensor([1, 2, 3])
max = torch.argmax(tens)
print (max)