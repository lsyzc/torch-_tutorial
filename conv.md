# 卷积
## Conv2d 互相关
其实是互相关运算，相比真正卷积，少了卷积核旋转180度

conv2d(in_channel,out_channel,(kernel_size,kernel_size))

一个 in_channel * kernel_size * kernel_size 操作后得到 1 * H * W

out_channel 个卷积核concat得到 out_channel * H * W

padding controls the amount of padding applied to the input. It can be either a string {‘valid’, ‘same’} or an int / a tuple of ints giving the amount of implicit padding applied on both sides.

其中padding参数：
valid:不padding, same: 通过自动调整padding使得输出尺寸和输入一致

## 空洞卷积

## 分组卷积
conv2d(c,f,(kernel_size,kernel_size),groups = g)
可减少参数量,参数量为 kernel_size * kernel_size * c * f / g
conv2d(c,f,(kernel_size,kernel_size)) 参数量为 kernel_size * kernel_size * c * f 
具体操作video https://www.bilibili.com/video/BV1b9CHYbEtK?t=432.4
## 深度可分离卷积，groups = in_channels
