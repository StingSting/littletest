#!/usr/bin/env python
# coding: utf-8

# # Pytorch Tutorial

# Pytorch is a popular deep learning framework and it's easy to get started.

# In[ ]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time

BATCH_SIZE = 128
NUM_EPOCHS = 10


# First, we read the mnist data, preprocess them and encapsulate them into dataloader form.

# In[ ]:


# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)


# Then, we define the model, object function and optimizer that we use to classify.

# In[ ]:



# TODO:define model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)  # reshape
        out = self.fc(out)
        return out
model = SimpleNet()
# TODO:define loss function and optimiter

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),0.03)

# Next, we can start to train and evaluate!



for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch [%d/%d],  Accuracy: %.4f'% (epoch + 1, NUM_EPOCHS, 1-loss.item()))
 
# Save the Trained Model
torch.save(model.state_dict(), 'model.pkl')


correct=0.00
total=0.00
for data in test_loader:
    images,labels=data
    outputs=model(Variable(images))
    _,predicted=torch.max(outputs.data,1)
    total+=labels.size(0)
    correct+=(predicted==labels).sum()
print('Accuracy of test=%.4f'% (correct/total))
   
