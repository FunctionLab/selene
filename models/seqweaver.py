"""
Seqweaver architecture (Park & Troyanskaya, 2021).
"""
import torch
import torch.nn as nn


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class Seqweaver(nn.Module):

    def __init__(self, n_classes):  # 217 human, 43 mouse
        super(Seqweaver, self).__init__()
        self.model = nn.Sequential(
	        nn.Conv2d(4, 160, (1, 8)),
	        nn.ReLU(),
	        nn.MaxPool2d((1, 4), (1, 4)),
	        nn.Dropout(0.1),
	        nn.Conv2d(160, 320, (1, 8)),
	        nn.ReLU(),
	        nn.MaxPool2d((1, 4), (1, 4)),
	        nn.Dropout(0.1),
	        nn.Conv2d(320, 480, (1, 8)),
	        nn.ReLU(),
	        nn.Dropout(0.3))
        self.fc = nn.Sequential(
	        Lambda(lambda x: torch.reshape(x, (x.size(0), 25440))),
	        nn.Sequential(
                Lambda(lambda x: x.reshape(1, -1)
                       if 1 == len(x.size()) else x),
                nn.Linear(25440, n_classes)
            ),
	        nn.ReLU(),
	        nn.Sequential(
                Lambda(lambda x: x.view(1, -1)
                       if 1 == len(x.size()) else x),
                nn.Linear(n_classes, n_classes)
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.model(x)
        x = self.fc(x)
        return x


def criterion():
    return nn.BCELoss()


def get_optimizer(lr):
    return (torch.optim.SGD,
            {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})
