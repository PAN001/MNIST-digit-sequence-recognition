# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import*
import argparse
import os
import unidecode
import random
from tqdm import tqdm

argparser = argparse.ArgumentParser()
argparser.add_argument('file', type=str)
argparser.add_argument('--lr', type=float, default=0.01)
argparser.add_argument('--epochs', type=int, default=2000)
argparser.add_argument('--chunk_size', type=int, default=int(2000/20))
argparser.add_argument('--batch_size', type=int, default=100*20)
# argparser.add_argument('--cuda', type=bool, default=True)
args = argparser.parse_args()
args.cuda = False

file = unidecode.unidecode(open(args.file).read())

def save():
    savename = os.path.splitext(os.path.basename(args.file))[0] + '.pt'
    torch.save(net, savename)
    print('Saved')

def random_training_set():
    input = torch.LongTensor(args.batch_size, args.chunk_size)
    target = torch.LongTensor(args.batch_size, args.chunk_size)
    for bi in range(args.batch_size):
        start_index = random.randint(0, len(file) - args.chunk_size - 1)
        chunk = file[start_index: start_index + args.chunk_size + 1]

        # print "input: ", chunk[:-1]
        # print "target: ", chunk[1:]

        input[bi] = string2tensor(chunk[:-1])
        target[bi] = string2tensor(chunk[1:])
    return (Variable(input.cuda()), Variable(target.cuda())) if args.cuda else (Variable(input), Variable(target))

def train():
    net.reset_hidden()
    opt.zero_grad()
    loss = 0
    input, target = random_training_set()

    # 对于每个chunk，一个一个feed：
    # chunk: abc -> input: ab, target: bc
    # i = 0: input: a, target: b
    # i = 1: input: ab, target: bc
    for i in range(args.chunk_size):
        out = net(input[:,i]) # feed input
        # print "output: ", out.view(args.batch_size, -1)
        # print "target: ", target[:,i]
        loss += criterion(out.view(args.batch_size, -1), target[:,i])
    print "gd: ", loss.backward()
    opt.step()

net = CharRNN(args.cuda, args.batch_size)
if args.cuda: net = net.cuda()
opt = torch.optim.Adam(net.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

for i in tqdm(range(args.epochs)):
    # print i
    train()

save()
