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

x = np.zeros((3, 3))
x = torch.from_numpy(x)
x_back = x.numpy()
print(x_back)

# Addition
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
print(x + y)
z = torch.add(x, y)
z1 = torch.empty(3)
print(z1)
torch.add(x, y, out=z1)
print(z1)
# subtraction
z = x - y
# Division
z = torch.true_divide(x, y)
# inplace operation
t = torch.zeros(3)
t.add_(x)
t += x  # inplace
# t = t + x # create a new one
print(t)
# exponentiation
z = x.pow(2)
z = x**2
# simple comparsion
z = x > 0
print(z)
# matrix mutiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 2))
print(x1.shape)
z2 = torch.mm(x1, x2)  # z2 = torch.mul(x1,x2) 注释的是按元素相乘，会广播

z3 = x1.mm(x2)
print(x1)
print(x2)
# matrix exponentiation
matrix_one = torch.rand((5, 5))
mmatrix_power = matrix_one.matrix_power(3)
print(mmatrix_power)

# x = torch.rand((3,5,5))
x = torch.arange(75).reshape((3, 5, 5))
print(x.size())
y = torch.sum(x, dim=0)
print(y.shape)
print(x)
print(y)
x = torch.arange(5)
print(x)

print(type(torch.max(x, dim=0)))
max_item, index = torch.max(x, dim=0)
print(type(max_item))
item = torch.max(x, dim=0)
print(item)
x = torch.arange(5)
print(x)
print(
    type(torch.max(x, dim=0))
)  # torch.max / torch.min只要制定了dim 返回的结果就是一个tuple 包含values 和 indices
values, index_t = torch.max(x, dim=0)
print(values)
print(index_t)
print(torch.argmax(x))
m = torch.mean(x.float())  # the input tensor, either of floating point or complex dtype
print(m)
y = torch.arange(1, 6)
y[0] = 0
print(type(y[0]))
print(x == y)
print(torch.equal(x, y))
x = torch.tensor([12, 10, 2, 1])
sorted_x = torch.sort(x, dim=0)
print(type(sorted_x))
y, i = sorted_x
print(type(y))
print(y)
print(i)
z = torch.clamp(
    x, min=0, max=10
)  # any number in blow min will be set to min and number above max will be set to 10
print(z)

x = torch.tensor([0, 1, 1, 0, 1])
z = torch.any(x, keepdim=True)  # 其中有一个为真就是真
print(z)
z2 = torch.all(x, keepdim=True)  # keep dim 输入与输出一个维度
print(z2)

# tensor indexing
batch_size = 10
features = 25
x = torch.rand((batch_size, features))
print(x.shape)
print(x[0].shape)
print(x[:, 0].shape)
print(x[:, 0].unsqueeze(dim=1).shape)
x[0, 0] = 100
print(x[1, 0] == 100)

# fancy indexing
x = torch.arange(10)
indices = [2, 4, 5]
print(x[indices])

x = torch.arange(9).reshape((3,3))
print(x)
rows = [0,1]
cols = [1,2]
print(x[rows,cols]) #返回的是x[0,1] and cols[1,2] 易错
# more advanced indexing
x = torch.arange(10)
print(x)
print(x[(x<2) | (x>8)])

print(x.remainder(2) == 0) # x.remainder(2) 是x模2

#other usefule oprations
print(torch.where(x > 5,x,x*2)) # x > 5 ? x:2*x
print(torch.tensor([0,0,1,1,2,4]).unique())

# view reshape unsqueeze suqeeze permute 
x = torch.rand(9)
print(x)
y = x.view(3,3) # view()要求内存是连续的否则会报错，可以配合contiguous()使用
print(y)
x = torch.rand(3,3)


def test(*size):
    return size
r = test(1,2)
print(r)
print(type(r))
r2 = test((1,2))
print(r2)
print(type(r2))
# rand函数可以接受一串数字 或一个元组列表 是通过重载实现的