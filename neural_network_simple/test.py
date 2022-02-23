from re import L
import torch
from model import *
from torch.utils.data import DataLoader
import time
import sys

epochs = int(sys.argv[0]) # number of rounds
data_num = int(sys.argv[1]) # batches of data >= 1

test_data = torch.load('testing.pt')
train_data = torch.load('training/train0.pt')
for i in range(1, data_num):
    t = torch.load('training/train' + str(i) +'.pt')
    train_data = torch.cat((train_data, t))

trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
testloader = DataLoader(test_data, batch_size=32)

start = time.time()
net  = Net()
i = train(net, trainloader, epochs = 100)
end = time.time()
cross, acc = test(net, testloader)


print('time: {}, accuracy {}'.format(end - start, acc))