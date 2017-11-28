import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse
import torch.nn.init as init

class Net(nn.Module):

    def __init__(self, use_cuda):
        super(Net, self).__init__()

        self.classes = 10 + 1
        self.use_cuda = use_cuda
        self.image_H = 36

        # CNN
        # conv1
        self.conv1_input_chanel = 1
        self.conv1_output_chanel = 32
        self.conv1_kernelsize = (3, 3)
        self.conv1_stride = (2, 2)
        self.conv1 = nn.Conv2d(self.conv1_input_chanel, self.conv1_output_chanel, self.conv1_kernelsize, self.conv1_stride)

        # initialization
        init.xavier_uniform(self.conv1.weight, gain=np.sqrt(2))
        init.constant(self.conv1.bias, 0.1)

        self.inception_input_chanel = self.conv1_output_chanel
        self.mixed = InceptionA(self.inception_input_chanel, pool_features=16)
        self.conv_H = 10

        # conv2
        self.conv2_input_chanel = 16
        self.conv2_output_chanel = 32
        self.conv2_kernelsize = (3, 3)
        self.conv2_stride = (2, 2)
        self.conv2 = nn.Conv2d(self.conv2_input_chanel, self.conv2_output_chanel, self.conv2_kernelsize, self.conv2_stride)

        # LSTM
        self.lstm_input_size = self.conv_H * self.conv2_output_chanel  # number of features = H * cnn_output_chanel = 32 * 32 = 1024
        self.lstm_hidden_size = 32
        self.lstm_num_layers = 2
        self.lstm_hidden = None
        self.lstm_cell = None

        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_hidden_size, self.lstm_num_layers, batch_first = True, bidirectional = True)
        # # initialization
        # init.xavier_uniform(self.lstm.weights, gain=np.sqrt(2))
        # init.constant(self.lstm.bias, 0.1)

        # FC: convert to 11-d probability vector
        self.fc_input_size = self.lstm_hidden_size * 2
        self.fc_output_size = self.classes
        self.fc = nn.Linear(self.fc_input_size, self.fc_output_size)
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
        print "after conv1: ", out.size()
        out = F.relu(out)
        out = self.mixed(out)
        print "after inception: ", out.size()
        out = self.conv2(out);
        print "after conv2: ", out.size()

        # reshape
        out = out.permute(0, 3, 2, 1) # D(out) = (batch_size, W, H, cnn_output_chanel)
        out.contiguous()
        out = out.view(batch_size, -1, self.lstm_input_size) # D(out) = (batch_size, seq_len, lstm_input_size) where seq_len = W, lstm_input_size = H * cnn_output_chanel

        # print "before LSTM: ", out.size()
        # LSTM
        out, self.lstm_hidden = self.lstm(out, (self.lstm_hidden, self.lstm_cell)) # D(out) = (batch_size, seq_len, hidden_size)

        # reshape
        out.contiguous()
        out = out.view(-1, self.fc_input_size) # D(out) = (batch_size * seq_len, hidden_size)

        # fc layer
        out = self.fc(out) # D(out) = (batch_size * seq_len, classes)
        out = self.softmax(out)

        return out

    def reset_hidden(self, batch_size):
        # reset hidden state for time 0
        h0 = torch.zeros(self.lstm_num_layers * 2, batch_size, self.lstm_hidden_size) # random init
        h0 = h0.cuda() if self.use_cuda else h0
        self.lstm_hidden = Variable(h0)

    def reset_cell(self, batch_size):
        # reset cell state for time 0
        c0 = torch.zeros(self.lstm_num_layers * 2, batch_size, self.lstm_hidden_size) # random init
        c0 = c0.cuda() if self.use_cuda else c0
        self.lstm_cell = Variable(c0)

class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 8, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(8, 16, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 24, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(24, 16, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(16, 16, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool] # concat
        return torch.cat(outputs, 1)

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)