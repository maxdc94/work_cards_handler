import torch
print(torch.version.cuda)      # versione CUDA vista da PyTorch
print(torch.cuda.is_available())  # dovrebbe tornare True
print(torch.cuda.get_device_name(0))  # dovrebbe stampare "GeForce GTX 1050"
