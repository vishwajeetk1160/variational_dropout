import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class VariationalDropout(nn.Module):
    def __init__(self, input_size, out_size, log_sigma2=-10, threshold=3):
        """
        :param input_size: An int of input size
        :param log_sigma2: Initial value of log sigma ^ 2.
               It is crusial for training since it determines initial value of alpha
        :param threshold: Value for thresholding of validation. If log_alpha > threshold, then weight is zeroed
        :param out_size: An int of output size
        """
        super(VariationalDropout, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.theta = Parameter(t.FloatTensor(input_size, out_size))
        self.bias = Parameter(t.Tensor(out_size))

        self.log_sigma2 = Parameter(t.FloatTensor(input_size, out_size).fill_(log_sigma2))

        self.reset_parameters()

        self.k = [0.63576, 1.87320, 1.48695]

        self.threshold = threshold

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)

        self.theta.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    @staticmethod
    def clip(input, to=8.):
        input = input.masked_fill(input < -to, -to)
        input = input.masked_fill(input > to, to)

        return input

    def kld(self, log_alpha):

        first_term = self.k[0] * F.sigmoid(self.k[1] + self.k[2] * log_alpha)
        second_term = 0.5 * t.log(1 + t.exp(-log_alpha))
        return (first_term - second_term - self.k[0]).sum() / (self.input_size * self.out_size)


    def forward(self, input, train):
        """
        :param input: An float tensor with shape of [batch_size, input_size]
        :return: An float tensor with shape of [batch_size, out_size] and negative layer-kld estimation
        """
        log_alpha = self.clip(self.log_sigma2 - t.log(self.theta ** 2))
        #fh=open("log_alpha_values_during_training.txt", 'a')
        #fh.write(str(self.input_size)+"-----"+str(log_alpha.data.numpy().mean())+"-----"+str(self.out_size)+"\n")
        #fh.close()
        kld = self.kld(log_alpha)

        if not train:
            mask = log_alpha > self.threshold
            if (t.nonzero(mask).dim()!= 0):
                zeroed_weights=t.nonzero(mask).size(0)
                
            else :
                zeroed_weights=0
                
            total_weights=mask.size(0)*mask.size(1)
            print('number of zeroed weights is {}'.format(zeroed_weights))
            print ('total numer of weights is {}'.format(total_weights))
            print ('ratio for non zeroed weights is {}'.format( (total_weights - zeroed_weights)/total_weights) )
            return t.addmm(self.bias, input, self.theta.masked_fill(mask, 0))

        mu = t.mm(input, self.theta)
        std = t.sqrt(t.mm(input ** 2, self.log_sigma2.exp()) + 1e-6)

        eps = Variable(t.randn(*mu.size()))
        if input.is_cuda:
            eps = eps.cuda()

        return std * eps + mu + self.bias, kld

    def max_alpha(self):
        log_alpha = self.log_sigma2 - self.theta ** 2
        return t.max(log_alpha)