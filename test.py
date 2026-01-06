import torch
print(torch.version.cuda)      # Cuda Version
print(torch.cuda.is_available())  
print(torch.cuda.get_device_name(0))  # video card name
