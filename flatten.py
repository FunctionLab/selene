import math

import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules import Module


class Flatten(Module):
    def __init__(self, in_features, out_features, n_genomic_features, regularization_dim=1, bias=True):
        """
        bias: the learnable bias of the module of shape (out_features)
        """
        super(Flatten, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = Parameter(torch.Tensor(n_genomic_features, in_features, regularization_dim))
        self.weight2 = Parameter(torch.Tensor(n_genomic_features, regularization_dim, out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        stdv2 = 1. / math.sqrt(self.weight2.size(2))
        self.weight1.data.uniform_(-stdv1, stdv1)
        self.weight2.data.uniform_(-stdv2, stdv2)
        if self.bias is not None:
            stdv_avg = (stdv1 + stdv2) / 2
            self.bias.data.uniform_(-stdv_avg, stdv_avg)

    def forward(self, input):
        #print("fwd flatten")
        #print(input.size())
        #print(type(input))
        #print(type(self.weight1))
        #print(type(self.weight2))
        #weight_matrix = torch.mm(self.weight1.data, self.weight2.data)
        weight_matrix = self.weight1.data.matmul(self.weight2.data)

        #weight_matrix = np.matmul(self.weight1.data, self.weight2.data)
        weight_matrix.transpose_(1, 2)

        #print("WM SIZE {0}".format(weight_matrix.size()))
        # flatten
        weight_matrix = weight_matrix.contiguous().view(weight_matrix.size(0), -1)

        weight_matrix_param = Parameter(weight_matrix)
        #print("FWD {0}".format(weight_matrix_param.size()))
        #weight_matrix_param = Parameter(torch.Tensor(weight_matrix))
        #input_t = input.transpose(1, 2)
        return F.linear(input, weight_matrix_param)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'
