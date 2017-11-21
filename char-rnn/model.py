# -*- coding: utf-8 -*-

# https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import string


chars = string.printable

# Turn a string into a vector with each char decoded as the index in string.printable
def string2tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = chars.index(string[c])
        except:
            continue
    return tensor
    #return torch.Tensor(list(map(lambda x: chars.index(x), string)))

class CharRNN(nn.Module):
    def __init__(self, cuda, batch_size, input_size=100, output_size=100):
        super(CharRNN, self).__init__()
        
        self.useCuda = cuda
        self.batch_size = batch_size
        self.input_size = input_size # the number of distinct characters in string.printable
        self.n_layers = 1 
        self.n_hidden = 100

        self.reset_hidden()

        # encodes the input character into an internal state
        # ???，word embedding具体怎么操作的？
        self.encode = nn.Embedding(input_size, self.n_hidden) # size of the dictionary of embeddings: 100, size of each embedding vector: n_hidden
        self.rnn = nn.GRU(self.n_hidden, self.n_hidden, self.n_layers)
        self.decode = nn.Linear(self.n_hidden, output_size)

    def forward(self, x):
        # the input is a sequence of characters (each character is presented by its index in string.printable

        e = self.encode(x).view(1, x.size(0), -1) # reshape

        out, self.hidden = self.rnn(e, self.hidden)
        return self.decode(out.view(self.batch_size, self.n_hidden))

    def reset_hidden(self):
        zs = torch.zeros(self.n_layers, self.batch_size, self.n_hidden)
        zs = zs.cuda() if self.useCuda else zs
        self.hidden = Variable(zs)

