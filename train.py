#KD part
import argparse
import torch as t
import torch.nn as nn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets
import matplotlib.pyplot as plt

from models import *

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

    #writer = SummaryWriter(args.mode)

    assert args.mode in ['simple', 'dropout', 'vardropout'], 'Invalid mode, should be in [simple, dropout, vardropout]'
    Model = {
        'simple': SimpleModel,
        'dropout': DropoutModel,
        'vardropout': VariationalDropoutModel
    }
    Model = Model[args.mode]

    dataset = datasets.MNIST(root='data/',
                             transform=transforms.Compose([
                                 transforms.ToTensor()]),
                             download=True,
                             train=True)
    train_dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    dataset = datasets.MNIST(root='data/',
                             transform=transforms.Compose([
                                 transforms.ToTensor()]),
                             download=True,
                             train=False)
    test_dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)


    model = Model()
    if args.use_cuda:
        model.cuda()

    optimizer = Adam(model.parameters(), args.learning_rate, eps=1e-6)
    coef1=1.
    coef2=0.275

    fh=open("training_results.txt", "a")
    fh.write("\n")
    fh.write("\n")
    fh.write("\n")

    fh.write("batch_size {}, epochs {}, learning rate {}, coeff kld {}, coeff kd{}".format(args.batch_size,args.num_epochs, args.learning_rate, coef1, coef2))
    fh.write("\n")

    cross_enropy_averaged = nn.CrossEntropyLoss(size_average=True)
    train_loss_list=[]
    validation_loss_list=[]
    for epoch in range(args.num_epochs):
        for iteration, (input, target) in enumerate(train_dataloader):
            input = Variable(input).view(-1, 784)
            target = Variable(target)
            
            if args.use_cuda:
                input, target = input.cuda(), target.cuda()

            optimizer.zero_grad()

            loss = None
            if args.mode == 'simple':
                loss = model.loss(input=input, target=target, average=True)
            elif args.mode == 'dropout':
                loss = model.loss(input=input, target=target, p=0.4, average=True)
            else:
                likelihood, kld, kd_loss_value = model.loss(input=input, target=target, train=True, average=True)
                #coef = min(epoch / 40., 1.) #making this ceff. 1. understand this coeff. 
                #print ("kd_loss_value")
                #print (kd_loss_value.data)
                loss = likelihood+coef1* kld+ coef2*kd_loss_value
                train_loss_list.append(loss.cpu().data.numpy()[0])

            loss.backward()
            optimizer.step()
            if iteration % 300 == 0:
            	print('train epoch {}, iteration {}, loss {}'.format(epoch, iteration, loss.cpu().data.numpy()[0]))
            	fh.write('train epoch {}, iteration {}, loss {}'.format(epoch, iteration, loss.cpu().data.numpy()[0]))
            	fh.write('\n')

            if (iteration % 300) == 0:
                loss = 0
                for input, target in test_dataloader:
                    input = Variable(input).view(-1, 784)
                    target = Variable(target)

                    if args.use_cuda:
                        input, target = input.cuda(), target.cuda()

                    if args.mode == 'simple':
                        loss += model.loss(input=input, target=target, average=False).cpu().data.numpy()[0]
                    elif args.mode == 'dropout':
                        loss += model.loss(input=input, target=target, p=0., average=False).cpu().data.numpy()[0]
                    else:
                        loss += model.loss(input=input, target=target, train=False, average=False).cpu().data.numpy()[0]

                loss = loss / (args.batch_size * len(test_dataloader))
                print('validation epoch {}, iteration {}, loss {}'.format(epoch, iteration, loss))
                fh.write('validation epoch {}, iteration {}, loss {}'.format(epoch, iteration, loss))
                fh.write('\n')
                
    t.save(model.state_dict(), 'trained_model_NewJittered_275.pth.tar')
    plt.figure(1)
    plt.plot(train_loss_list)
    plt.show()

fh.close()
