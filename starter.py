import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse

labels = torch.Tensor(np.load('./labels.npy'))
data = torch.Tensor(np.load('dataset/data.npy'))

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
args = parser.parse_args()

# See http://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
class CTC(torch.autograd.Function):
    def forward(self, input):
        self.save_for_backward(input)
        return input

    def backward(self, grad_output):
        return grad_output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        return x

model = Net()

torch.save(model, 'TrainedModel')
