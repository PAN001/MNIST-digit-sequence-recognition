# -*- coding: utf-8 -*-

from model import *
from CTCLoss import *

import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument('--lr', type=float, default=0.01)
argparser.add_argument('--epochs', type=int, default=2000)
argparser.add_argument('--batch_size', type=int, default=100*20)
argparser.add_argument('--cuda', type=bool, default=False)
args = argparser.parse_args()
args.cuda = False

batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
classes = 10 + 1
cuda = args.cuda

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

    # plt.title(str(predictions[0]))
    # plt.imshow(dataset_data[0], cmap='gray')
    # plt.show()

    print "loss: ", loss

    predictions = criterion.decode_best_path(out)
    print "best_path_predictions[0]: ", predictions[0]

    # pred_0, score_0 = criterion.decode_beam(out.data.numpy()[0])
    # print "beam_predictions[0]: ", pred_0

    print "label[0]: ", target.data.numpy()[0]

    loss.backward()
    opt.step()

net = Net(cuda, batch_size)
opt = torch.optim.Adam(net.parameters(), lr = lr)
# opt = optim.SGD(net.parameters(), lr = lr, momentum=0.9)
criterion = CTCLoss()

dataset_data = np.load("./dataset/data_5_10000.npy")
dataset_labels = np.load("./dataset/labels_5_10000.npy")
dataset_labels = dataset_labels.astype(int)

data = torch.Tensor(dataset_data)
shape = data.shape
data = data.view(dataset_data.shape[0], 1, dataset_data.shape[1], dataset_data.shape[2])
data = Variable(data.cuda()) if cuda else Variable(data)
labels = torch.IntTensor(dataset_labels)
labels = Variable(labels.cuda()) if cuda else Variable(labels)

for i in tqdm(range(epochs)):
    print ""
    data_batch = data[i * batch_size: (i+1) * batch_size]
    labels_batch = labels[i * batch_size: (i+1) * batch_size]
    train(data_batch, labels_batch)

