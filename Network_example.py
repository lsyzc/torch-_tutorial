# Import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# create fully connected Network
class Net(nn.Module):
    def __init__(self, num_size, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features=num_size, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set hyperparameters
num_epochs = 5
input_size = 784
num_class = 10
l_rate = 0.001
batch_size = 64
print(device)
# load data
train_dataset = datasets.MNIST(root="./MNIST",train=True,transform=transforms.ToTensor(),download=True)
train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataset = datasets.MNIST(root="./MNIST",train=False,transform=transforms.ToTensor(),download=True)
test_dataloader = DataLoader(dataset=test_dataset,shuffle=True,batch_size=batch_size)

# initialize network
model = Net(num_size=input_size,n_classes=num_class).to(device)
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(),lr=l_rate)
# train
for epoch in range(num_epochs):
    for data,target in train_dataloader:
        data = data.to(device)
        target = target.to(device)
        data = data.reshape(data.shape[0],-1)   # 这里不能写batch_size 因为最后一个epoch可能不够一整个batch_size,应该写成data.shape[0]
        predict = model(data)
        loss = criterion(predict,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def check_accuracy(loader,model):
    num_correct = 0
    num_samples = 0
    model = model.eval()

    for data,target in loader:
        data = data.to(device)
        target = target.to(device)
        data = data.reshape(data.shape[0],-1)
        with torch.no_grad():
            predict = model(data)
        predict_y = predict.argmax(1) # 找到预测值最大的索引 也就是对应的预测数字是多少
        print(predict_y)
        num_correct += (predict_y == target).sum().item()
        num_samples += target.size(0)
    result = num_correct / num_samples
    print(f"Accuracy: {result}")
    return result
        
check_accuracy(train_dataloader,model)
check_accuracy(test_dataloader,model)



