import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.tensor([1, 2, 3])
y = torch.tensor([7, 8, 9])

#Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)

z2 = torch.add(x, y)

z = x + y
print(z)

#Subtraction
z_sub = x - y
print(z_sub)

#Division
z_div = torch.true_divide(x, y)
print(z_div)

#inplace operations
t = torch.zeros(3)
t.add_(x)
print(t)
t+=x
print(t)

#Exponent
z = x.pow(2)
print(z)
z = x**2
print(z)

#Simple comparision
z=x<0
print(z)

#Matrix Mutiplication
x1 = torch.rand((2, 5))
print(x1)
x2 = torch.rand((5, 3))

x3 = torch.mm(x1, x2)
print(x3)

#Matrix exponentiation
matrix_exp = torch.rand(5,5)
print(matrix_exp.matrix_power(4))

#Element wise multiplication
z = x*y
print(z)

#Batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1= torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)
print(out_bmm)

#Example of Braodcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2
z = x1** x2

#Other useful tensor operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)

abs_x = torch.abs(x)
z = torch.argmax(x, dim= 0)




