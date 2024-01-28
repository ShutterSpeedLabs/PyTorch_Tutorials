import torch


#-------------- Tensor initialization -----------#

device = "cuda" if torch.cuda.is_available() else "cpu"

test_tensor = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 14]], dtype=torch.float32, device=device, requires_grad=True)

print(test_tensor)
print(test_tensor.shape)
print(test_tensor.device)
print(test_tensor.dtype)
print(test_tensor.requires_grad)
