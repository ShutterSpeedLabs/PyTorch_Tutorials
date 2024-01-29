import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

#-------------- Tensor initialization -----------#

mat_x = torch.empty(size=(5, 5), dtype=torch.float32, device=device)            #Create a empty tensor of size 5x5 
print(mat_x)
mat_x = torch.zeros((4, 4), dtype=torch.int64, device=device)                   #Create a tensor with values zero of size 4x4
print(mat_x)
mat_x = torch.rand(5,5)                                                         #Create a tensor of size 5x5 random values between 0-1
print(mat_x)
mat_x = torch.eye(5)                                                            #Create identity matrix of 5x5
print(mat_x)
mat_x = torch.ones(4,4)                                                         #Create a matrix of 4x4 all elements having value 1
print(mat_x)
mat_x = torch.arange(start=0, end=5, step=1)                                    #Create an tensor of values starting from 0 and ends at 5, step size is one
print(mat_x)
mat_x = torch.linspace(start=0.1, end=1, steps=10)                              #Create a tensor of 10 values equally divided between 0.1 and 1
print(mat_x)
mat_x = torch.diag(torch.ones(5))                                               #Create a matrix of 5x5 having diagonal elements as one remaining element zero
print(mat_x)
mat_x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
print(mat_x)
mat_x = torch.empty(size=(1, 5)).uniform_(0, 1)
print(mat_x)