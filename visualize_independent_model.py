import argparse
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--num-epochs', type=int, default=20, metavar='NI',
                        help='num epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='BS',
                        help='batch size (default: 70)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--mode', type=str, default='vardropout', metavar='M',
                        help='training mode (default: simple)')
    args = parser.parse_args()

    dataset = datasets.MNIST(root='data/',
                             transform=transforms.Compose([
                                 transforms.ToTensor()]),
                             download=True,
                             train=True)
    train_loader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    dataset = datasets.MNIST(root='data/',
                             transform=transforms.Compose([
                                 transforms.ToTensor()]),
                             download=True,
                             train=False)
    test_loader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 1200)
            self.fc2 = nn.Linear(1200, 1200)
            self.fc3 = nn.Linear(1200, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=0.8, training=self.training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=0.8, training=self.training)
            x = self.fc3(x)
            return x




    model = Model()
    if args.use_cuda:
        model.cuda()

    model.load_state_dict(t.load('teacher_MLP_jittered_Adam_try_2.pth.tar'))
    test_error=[]

    def test(model):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if args.use_cuda:
                data, target = data.cuda(), target.cuda()
            data = Variable(data, volatile=True).view(-1, 784)
            target = Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        test_error.append(100.0-float (100. * float(correct) / float(len(test_loader.dataset))))
        print('\nTest set: Average loss: {:.7f}, Accuracy: {}/{} ({:.7f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            float (100. * float(correct) / float(len(test_loader.dataset)))))

    test(model)
