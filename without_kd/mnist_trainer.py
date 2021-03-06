#teacher jittered model dimension and forwards stored
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.autograd import Variable
import os
import time
import random
start_time = time.time()
#epochs=50
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def jitter(data_tensor):
    for i in range(1, len(data_tensor[0]-1)):
        for j in range(1, len(data_tensor[0]-1)):
            random_int=random.randint(1,9)   
            var_1=data_tensor[0][i][j]
            if random_int == 1:
                data_tensor[0][i][j]=data_tensor[0][i-1][j-1]
                data_tensor[0][i-1][j-1]=var_1
            if random_int == 2:
                data_tensor[0][i][j]=data_tensor[0][i-1][j]
                data_tensor[0][i-1][j]=var_1
            if random_int == 3:
                data_tensor[0][i][j]=data_tensor[0][i-1][j+1]
                data_tensor[0][i-1][j+1]=var_1
            if random_int == 4:
                data_tensor[0][i][j]=data_tensor[0][i][j-1]
                data_tensor[0][i][j-1]=var_1
            if random_int == 5:
                data_tensor[0][i][j]=data_tensor[0][i][j+1]
                data_tensor[0][i][j+1]=var_1
            if random_int == 6:
                data_tensor[0][i][j]=data_tensor[0][i+1][j-1]
                data_tensor[0][i+1][j-1]=var_1
            if random_int == 7:
                data_tensor[0][i][j]=data_tensor[0][i+1][j]
                data_tensor[0][i+1][j]=var_1
            if random_int ==8:
                data_tensor[0][i][j]=data_tensor[0][i+1][j+1]
                data_tensor[0][i+1][j+1]=var_1


data_train = MNIST(root='data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
data_test = MNIST (root='data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

for i in range(len(data_train)):
    jitter(data_train[i])


train_loader = torch.utils.data.DataLoader(dataset = data_train, batch_size = args.batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset= data_test, batch_size = args.test_batch_size, shuffle= True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.fc3(x)
        return x

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-8,
                      weight_decay=5e-4)

def train(epoch, model):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def train_evaluate(model):
    model.eval()
    train_loss = 0
    correct = 0
    for data, target in train_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        train_loss += F.cross_entropy(output, target).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
for epoch in range(1, args.epochs + 1):
    train(epoch, model)
    train_evaluate(model)
    test(model)

torch.save(model.state_dict(), 'student_without_ard_without_kd.pth.tar')
print("--- %s seconds ---" % (time.time() - start_time))
