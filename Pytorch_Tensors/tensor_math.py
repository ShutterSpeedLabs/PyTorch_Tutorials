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
