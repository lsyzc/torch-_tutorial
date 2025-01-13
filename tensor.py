import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor(
    [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True
)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)
# other comman initialization methods
x = torch.empty(size=(3, 3))
x = torch.zeros((3, 3))
x = torch.rand((3, 3))
print(x)
x = torch.ones((3, 3))
x = torch.eye(5, 5)
print(x)
x = torch.arange(start=0, end=3)
print(x)
x = torch.ones((3, 3))
print(x)
x = torch.diag(my_tensor)
print(x)


# How to initialize and convert tensors to other types(int float double)
tensor = torch.arange(4)
print(tensor.bool())
print(tensor.float())
print(tensor.double())
print(tensor.int())
print(tensor.half())

# how to convert a numpy to a tensor

import numpy as np

x = np.zeros((3,3))
x = torch.from_numpy(x)
x_back = x.numpy()
print(x_back)

#Addition
x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])
print(x + y)
z = torch.add(x,y)
z1 = torch.empty(3)
print(z1)
torch.add(x,y,out=z1)
print(z1)
# subtraction
z = x - y
# Division
z = torch.true_divide(x,y)
# inplace operation
t = torch.zeros(3)
t.add_(x)
t += x # inplace
#t = t + x # create a new one
print(t)
# exponentiation
z = x.pow(2)
z = x**2
# simple comparsion
z = x > 0
print(z)
# matrix mutiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,2))
print(x1.shape)
z2 = torch.mm(x1,x2)#z2 = torch.mul(x1,x2) 注释的是按元素相乘，会广播

z3 = x1.mm(x2)
print(x1)
print(x2)
# matrix exponentiation
matrix_one = torch.rand((5,5))
mmatrix_power = matrix_one.matrix_power(3)
print(mmatrix_power)

#x = torch.rand((3,5,5))
x = torch.arange(75).reshape((3,5,5))
print(x.size())
y = torch.sum(x,dim=0)
print(y.shape)
print(x)
print(y)
