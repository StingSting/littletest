#!/usr/bin/env python
# coding: utf-8

# # Pytorch Tutorial

# Pytorch is a popular deep learning framework and it's easy to get started.

# In[ ]:


import torch
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


class SimpleNet(nn.Module):
# TODO:define model
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

model = SimpleNet(784,200,100,10)

# TODO:define loss function and optimiter

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),0.03)

# Next, we can start to train and evaluate!

# In[ ]:


# train and evaluate
for epoch in range(NUM_EPOCHS):
    for images, labels in tqdm(train_loader):
        # TODO:forward + backward + optimize
        img, label = iter(train_loader).next()
        if torch.cuda.is_available():
            img = Variable(img.view(img.size(0), -1)).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img.view(img.size(0), -1))
            label = Variable(label)
        #前向传播
        out = model(img)
        loss = criterion(out, label)
        #反向传播
        optimizer.zero_grad()#梯度归零
        loss.backward()
        optimizer.step()#更新参数
        if(epoch + 1) % 100 == 0:
            print('*'*10)
            print('epoch{}'.format(epoch+1))
            print('loss is {:.4f}'.format(loss.item()))
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss:{:.6f}, Acc:{:.6f}'.format(eval_loss / (len(test_dataset)), eval_acc / (len(test_dataset))))
        
    # evaluate
    # TODO:calculate the accuracy using traning and testing dataset
    
    
    
    


# #### Q5:
# Please print the training and testing accuracy.
