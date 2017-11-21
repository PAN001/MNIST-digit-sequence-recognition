# -*- coding: utf-8 -*-

from model import *
from CTCLoss import *

import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# argparser = argparse.ArgumentParser()
# argparser.add_argument('file', type=str)
# argparser.add_argument('--lr', type=float, default=0.01)
# argparser.add_argument('--epochs', type=int, default=2000)
# argparser.add_argument('--window_size', type=int, default=int(2000/20))
# argparser.add_argument('--batch_size', type=int, default=100*20)
# argparser.add_argument('--cuda', type=bool, default=True)
# args = argparser.parse_args()
# args.cuda = False

batch_size = 100
lr = 0.01
epochs = 100
input_length = 50
classes = 10 + 1
cuda = False
seq_length = 20
#
# def random_training_set():
#     input = torch.LongTensor(batch_size, chunk_size)
#     target = torch.LongTensor(batch_size, chunk_size)
#     # for bi in range(args.batch_size):
#     #     start_index = random.randint(0, len(file) - args.chunk_size - 1)
#     #     chunk = file[start_index: start_index + args.chunk_size + 1]
#     #
#     #     print "input: ", chunk[:-1]
#     #     print "target: ", chunk[1:]
#     #
#     #     input[bi] = string2tensor(chunk[:-1])
#     #     target[bi] = string2tensor(chunk[1:])
#     # return (Variable(input.cuda()), Variable(target.cuda())) if args.cuda else (Variable(input), Variable(target))

def train(input, target):
    net.reset_hidden()
    opt.zero_grad()
    loss = 0

    # # splice
    # cnt = 0
    # outs = []
    # while cnt < shape[2]:
    #     # print input[:,:,:,cnt:cnt + window_size]
    #     out = net(input[:,:,:,cnt:cnt + window_size]) # feed input
    #     out = out.view(batch_size, -1)
    #     outs.append(out.numpy())
    #     cnt + window_size

    out = net(input) # D(out) = (batch_size * seq_len, classes)
    out = out.view(batch_size, -1, classes) # D(out) = (batch_size, seq_len, classes)
    loss = criterion(out, target)
    predictions = criterion.decode_best_path(out)

    # plt.title(str(predictions[0]))
    # plt.imshow(dataset_data[0], cmap='gray')
    # plt.show()

    print "loss: ", loss
    print "predictions[0]: ", predictions[0]
    loss.backward()
    opt.step()

net = Net(batch_size)
opt = torch.optim.Adam(net.parameters(), lr = lr)
criterion = CTCLoss()

dataset_data = np.load("./dataset/data_20.npy")
dataset_labels = np.load("./dataset/labels_20.npy")
dataset_labels = dataset_labels.astype(int)

data = torch.Tensor(dataset_data)
shape = data.shape
data = data.view(dataset_data.shape[0], 1, dataset_data.shape[1], dataset_data.shape[2])
data = Variable(data.cuda()) if cuda else Variable(data)
labels = torch.IntTensor(dataset_labels)
labels = Variable(labels.cuda()) if cuda else Variable(labels)

for i in tqdm(range(epochs)):
    print i
    train(data, labels)

