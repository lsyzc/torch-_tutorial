import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import torch.optim as optim
from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self,in_channels=1,class_num=10):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=8,
                               kernel_size=3,
                               padding=1,
                               stride=1
                               )
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.conv2 = nn.Conv2d(in_channels=8,
                               out_channels=16,
                               kernel_size=3,
                               padding=1,
                               stride=1)
        self.fc1 = nn.Linear(16*7*7,class_num)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1) # import
        x = self.fc1(x)
        return x
    
        
# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learn_rate = 0.001
batch_size = 64
in_channels = 1
num_class = 10

train_dataset = datasets.MNIST(root="./MNIST",train=True,transform=transforms.ToTensor(),download=False)
test_dataset = datasets.MNIST(root="./MNIST",train=False,transform=transforms.ToTensor(),download=False)

train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

#init model
model = CNN(in_channels=in_channels,class_num=num_class)




        