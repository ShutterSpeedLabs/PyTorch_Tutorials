import torch
import numpy as np

#-------------- Tensor initialization -----------#

device = "cuda" if torch.cuda.is_available() else "cpu"

tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.float())
print(tensor.double())

np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)

np_array_new = tensor.numpy()