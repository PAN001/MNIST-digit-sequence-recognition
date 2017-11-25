import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse
import torch.nn.init as init

# # custom weights initialization
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)

class Net(nn.Module):

    def __init__(self, use_cuda):
        super(Net, self).__init__()

        self.classes = 10 + 1
        self.use_cuda = use_cuda
        self.image_H = 36

        # CNN
        # conv1
        self.conv1_input_chanel = 1
        self.conv1_output_chanel = 10
        self.conv1_kernelsize = (self.image_H, 5)
        self.conv1 = nn.Conv2d(self.conv1_input_chanel, self.conv1_output_chanel, self.conv1_kernelsize)

        # initialization
        init.xavier_uniform(self.conv1.weight, gain=np.sqrt(2))
        init.constant(self.conv1.bias, 0.1)

        # maxpool1
        self.maxpool1_kernelsize = (1,2)
        self.maxpool1 = nn.MaxPool2d(self.maxpool1_kernelsize, stride=1)

        # conv2
        self.conv2_input_chanel = 10
        self.conv2_output_chanel = 20
        self.conv2_kernelsize = (1, 5)
        self.conv2 = nn.Conv2d(self.conv2_input_chanel, self.conv2_output_chanel, self.conv2_kernelsize)

        # initialization
        init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2))
        init.constant(self.conv2.bias, 0.1)

        # maxpool2
        self.maxpool2_kernelsize = (1,2)
        self.maxpool2 = nn.MaxPool2d(self.maxpool2_kernelsize, stride=1)

        # batch norm (before activation)
        self.conv2_bn = nn.BatchNorm2d(self.conv2_output_chanel) # batch normalization

        # drop out (after activation)
        self.conv2_drop = nn.Dropout2d()

        self.conv2_H = 1 # height of feature map after conv2

        # LSTM
        self.lstm_input_size = self.conv2_H * self.conv2_output_chanel  # number of features = H * cnn_output_chanel = 32 * 32 = 1024
        self.lstm_hidden_size = 32
        self.lstm_num_layers = 1
        self.lstm_hidden = None
        self.lstm_cell = None

        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_hidden_size, self.lstm_num_layers, batch_first = True)
        # # initialization
        # init.xavier_uniform(self.lstm.weights, gain=np.sqrt(2))
        # init.constant(self.lstm.bias, 0.1)

        # FC: convert to 11-d probability vector
        self.fc_output_size = self.classes
        self.fc = nn.Linear(self.lstm_hidden_size, self.fc_output_size)
        # initialization
        init.xavier_uniform(self.fc.weight, gain=np.sqrt(2))
        init.constant(self.fc.bias, 0.1)

        # softmax:
        self.softmax = nn.Softmax()


    def forward(self, x):
        """
        Arguments:
            x: D

        """

        # CNN
        # print "input size: ", x.size()
        batch_size = int(x.size()[0])
        out = self.conv1(x) # D(out) = (batch_size, cov1_output_chanel, H, W)
        out = self.maxpool1(out)
        out = F.relu(out)
        # print "after conv1: ", out.size()

        out = self.conv2(out)
        out = self.maxpool2(out)
        out = self.conv2_bn(out) # bn before activation
        out = F.relu(out)
        out = self.conv2_drop(out) # drop after activation
        # print "after conv2: ", out.size()

        # reshape
        out = out.permute(0, 3, 2, 1) # D(out) = (batch_size, W, H, cnn_output_chanel)
        out.contiguous()
        out = out.view(batch_size, -1, self.lstm_input_size) # D(out) = (batch_size, seq_len, lstm_input_size) where seq_len = W, lstm_input_size = H * cnn_output_chanel

        # print "before LSTM: ", out.size()
        # LSTM
        out, self.lstm_hidden = self.lstm(out, (self.lstm_hidden, self.lstm_cell)) # D(out) = (batch_size, seq_len, hidden_size)

        # reshape
        out.contiguous()
        out = out.view(-1, self.lstm_hidden_size) # D(out) = (batch_size * seq_len, hidden_size)

        # fc layer
        out = self.fc(out) # D(out) = (batch_size * seq_len, classes)
        out = self.softmax(out)

        return out

    def reset_hidden(self, batch_size):
        # reset hidden state for time 0
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size) # random init
        h0 = h0.cuda() if self.use_cuda else h0
        self.lstm_hidden = Variable(h0)

    def reset_cell(self, batch_size):
        # reset cell state for time 0
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size) # random init
        c0 = c0.cuda() if self.use_cuda else c0
        self.lstm_cell = Variable(c0)