from re import L
import torch
from model import *
from torch.utils.data import DataLoader
import time

test_data = torch.load('base/test.pt')
train_data = torch.load('base/train0.pt')
for i in range(1, 4):
    t = torch.load('base/train' + str(i) +'.pt')
    train_data = torch.cat((train_data, t))

trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
testloader = DataLoader(test_data, batch_size=32)

start = time.time()
net  = Net()
i = train(net, trainloader, epochs = 100)
end = time.time()
cross, acc = test(net, testloader)


print('time: {}, accuracy {}'.format(end - start, acc))