import torch.nn as nn
import torch.nn.functional as F
import torch as t
from variational_dropout.variational_dropout import VariationalDropout

class teacherNet(nn.Module):
    def __init__(self):
        super(teacherNet, self).__init__()
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

class VariationalDropoutModel(nn.Module):
    def __init__(self):
        super(VariationalDropoutModel, self).__init__()

        self.fc = nn.ModuleList([
            VariationalDropout(784, 400),
            VariationalDropout(400, 40),
            nn.Linear(40, 10)
        ])

    def forward(self, input, train=False):
        """
        :param input: An float tensor with shape of [batch_size, 784]
        :param train: An boolean value indicating whether forward propagation called when training is performed
        :return: An float tensor with shape of [batch_size, 10]
                 filled with logits of likelihood and kld estimation
        """
        result = input

        if train:
            kld = 0

            for i, layer in enumerate(self.fc):
                if i != len(self.fc) - 1:
                    result, kld1 = layer(result, train)
                    result = F.elu(result)
                    kld =kld + kld1

            return self.fc[-1](result), kld

        for i, layer in enumerate(self.fc):
            if i != len(self.fc) - 1:
                result = F.elu(layer(result, train))

        return self.fc[-1](result)

    def loss(self, **kwargs):
        if kwargs['train']:
            input1=kwargs['input']
            out, kld = self(input1, train=kwargs['train'])
            return F.cross_entropy(out, kwargs['target'], size_average=kwargs['average']), kld

        out = self(kwargs['input'], kwargs['train'])
        return F.cross_entropy(out, kwargs['target'], size_average=kwargs['average'])

