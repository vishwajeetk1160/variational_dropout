import torch.nn as nn
import torch.nn.functional as F

from variational_dropout.variational_dropout import VariationalDropout


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

            return F.softmax(self.fc[-1](result)), kld

        for i, layer in enumerate(self.fc):
            if i != len(self.fc) - 1:
                result = F.elu(layer(result, train))

        return F.softmax(self.fc[-1](result))

    def loss(self, **kwargs):
        if kwargs['train']:
            input1=kwargs['input']
            out, kld = self(input1, train=kwargs['train'])
            #print('kld loss {}'.format(kld.data))
            #print('corss entropy loss {}'.format(F.cross_entropy(out, kwargs['target']).data))
            return F.cross_entropy(out, kwargs['target'], size_average=kwargs['average']), kld

        out = self(kwargs['input'], kwargs['train'])
        return F.cross_entropy(out, kwargs['target'], size_average=kwargs['average'])
