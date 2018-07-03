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

            return F.softmax(self.fc[-1](result)), kld

        for i, layer in enumerate(self.fc):
            if i != len(self.fc) - 1:
                result = F.elu(layer(result, train))

        return F.softmax(self.fc[-1](result))

    def loss(self, **kwargs):
        if kwargs['train']:
            input1=kwargs['input']
            out, kld = self(input1, train=kwargs['train'])
            teacher_model = teacherNet()
            teacher_model.load_state_dict(t.load('teacher_MLP_epch_30.pth.tar'))
            teacher_output=teacher_model(input1)
            #teacher_output= teacher_output.detach()
            kd_loss_value=kd_loss(out, teacher_output,2.)

            return F.cross_entropy(out, kwargs['target'], size_average=kwargs['average']), kld, kd_loss_value

        out = self(kwargs['input'], kwargs['train'])
        return F.cross_entropy(out, kwargs['target'], size_average=kwargs['average'])

    def kd_loss(self, y, teacher_scores, T):
        return nn.KLDivLoss()(F.log_softmax(y/T), F.softmax(teacher_scores/T)) * (T*T * 2.0)


