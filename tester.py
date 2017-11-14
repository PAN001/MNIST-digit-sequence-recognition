import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

labels = torch.Tensor(np.load('dataset/labels.npy'))
data = torch.Tensor(np.load('dataset/data.npy'))

model = torch.load('TrainedModel')

def test(epoch):
    test_loss = 0
    correct = 0
    total = 0

    inputs, targets = data, labels
    if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs, volatile=True), Variable(targets)

    outputs = net(inputs)
    loss = criterion(outputs, targets)

    test_loss += loss.data[0]
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()

    print('\nTest set: Average loss: {:.4f} Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data), 100. * correct / len(data))
