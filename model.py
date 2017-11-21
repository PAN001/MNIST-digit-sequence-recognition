import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse


class Net(nn.Module):

    def __init__(self, use_cuda, batch_size):
        super(Net, self).__init__()

        self.classes = 10 + 1
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.image_H = 32

        # CNN
        self.cnn_input_chanel = 1
        self.cnn_output_chanel = 32
        self.cnn_conv_kernelsize = 5
        self.pool_kernelsize = 2
        # self.pool_stride = 2

        self.conv = nn.Conv2d(self.cnn_input_chanel, self.cnn_output_chanel, self.cnn_conv_kernelsize)
        # self.pool = nn.MaxPool2d(self.pool_kernelsize, self.pool_stride)

        # LSTM
        self.lstm_input_size = self.image_H * self.cnn_output_chanel  # number of features = H * cnn_output_chanel = 32 * 32 = 1024
        self.lstm_hidden_size = 100
        self.lstm_num_layers = 1

        self.lstm_hidden = None
        self.lstm_cell = None

        self.reset_hidden()
        self.reset_cell()
        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_hidden_size, self.lstm_num_layers, batch_first = True)

        # MLP: convert to 11-d probability vector
        self.mlp_output_size = self.classes
        self.mlp = nn.Linear(self.lstm_hidden_size, self.mlp_output_size)

        # softmax:
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.conv(x) # D(out) = (batch_size, cnn_output_chanel, H, W)
        out = F.relu(out)

        # print "D(out) = (batch_size, cnn_output_chanel, H, W): ", out.size()
        out = out.permute(0, 3, 2, 1) # D(out) = (batch_size, W, H, cnn_output_chanel)
        out.contiguous()
        out = out.view(self.batch_size, -1, self.lstm_input_size) # D(out) = (batch_size, seq_len, lstm_input_size) where seq_len = W, lstm_input_size = H * cnn_output_chanel
        # print "D(out) = (batch_size, seq_len, lstm_input_size): ", out.size()
        out, self.lstm_hidden = self.lstm(out, (self.lstm_hidden, self.lstm_cell)) # D(out) = (batch_size, seq_len, hidden_size)
        # print "D(out) = (batch_size, seq_len, hidden_size): ", out.size()

        out.contiguous()
        out = out.view(-1, self.lstm_hidden_size) # D(out) = (batch_size * seq_len, hidden_size)
        # print "D(out) = (batch_size * seq_len, hidden_size): ", out.size()
        out = self.mlp(out) # D(out) = (batch_size * seq_len, classes)
        out = self.softmax(out)
        return out

    def reset_hidden(self):
        # reset hidden state for time 0
        h0 = torch.zeros(self.lstm_num_layers, self.batch_size, self.lstm_hidden_size) # random init
        h0 = h0.cuda() if self.use_cuda else h0
        self.lstm_hidden = Variable(h0)

    def reset_cell(self):
        # reset cell state for time 0
        c0 = torch.zeros(self.lstm_num_layers, self.batch_size, self.lstm_hidden_size) # random init
        c0 = c0.cuda() if self.use_cuda else c0
        self.lstm_cell = Variable(c0)