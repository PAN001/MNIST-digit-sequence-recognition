# -*- coding: utf-8 -*-

import torch

from model import*

import os
import argparse
import string

def generate(net, prime_str='A', predict_len=1000, temp=1.0, cuda=True):
    net.batch_size = 1
    net.reset_hidden()
    prime_input = Variable(string2tensor(prime_str).unsqueeze(0))

    if cuda: prime_input = prime_input.cuda()
    predicted = prime_str

    for p in range(len(prime_str) - 1):
        _ = net(prime_input[:,p])

    inp = prime_input[:,-1]

    for p in range(predict_len):
        output = net(inp)
        print output

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temp).exp() # softmax on the output (without normalization)
        top_i = torch.multinomial(output_dist, 1)[0]

        predicted_char = string.printable[top_i]
        predicted += predicted_char
        inp = Variable(string2tensor(predicted_char).unsqueeze(0))
        if cuda: inp = inp.cuda()

    return predicted

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('file', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type =int, default=1000)
    argparser.add_argument('-t', '--temp', type=float, default=1.0)
    # argparser.add_argument('--cuda', type=bool, default=True)
    args = argparser.parse_args()
    args.cuda = False

    net = torch.load(args.file)
    del args.file
    print(generate(net, **vars(args)))
