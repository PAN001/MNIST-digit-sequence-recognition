# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import numpy as np
dtype = torch.FloatTensor
import torch.nn.functional as F
import math
import collections
import time
import cv2

NEG_INF = -float("inf")

class CTCLoss(torch.autograd.Function):
    def __init__(self, cuda):
        super(CTCLoss, self).__init__()
        self.cuda = cuda

    def forward(self, input, seqs, blank = 10):
        """
        CTC loss function

        Args:
            input: D = (batch_size, classes, seq_len): Output from network for each sample in a batch
            seqs: D = (batch_size, seq_len): Target labels for each sample in a batch

        Returns the accumulated sum in a batch.
        """

        self.blank = blank
        self.batch_size = input.shape[0]

        input_np = input.cpu().numpy() if self.cuda else input.numpy()
        seqs_np = seqs.cpu().numpy() if self.cuda else seqs.numpy()

        self.input_np = input_np
        self.seqs_np = seqs_np

        alphases = []
        ll_forwards = []

        sum = 0.0

        for i in range(0, input.shape[0]): # iterate over each training sample
            probs = input_np[i] # D = (classes, seq_len): matrix of classes-D probability distributions over seq_len timestamps
            seq = seqs_np[i] # D = (seq_len): sequence of features for given sample

            seq_len = seq.shape[0]  # length of label sequence (# expected digits)
            L = 2 * seq_len + 1  # length of label sequence with blanks. e.g. abc -> _a_b_c_
            T = probs.shape[1]  # length of input sequence (time) (m)

            # forward dynamic table: L * T
            # alphas[u, t] represent the sum of probability of all paths outputing l'u and time t
            alphas = np.zeros((L, T))

            # initialize alphas and forward pass
            alphas[0, 0] = probs[self.blank, 0]  # a(u, t)
            alphas[1, 0] = probs[seq[0], 0]
            c = np.sum(alphas[:, 0])

            alphas[:, 0] = alphas[:, 0] / c
            ll_forward = np.log(c) # log liklihood of forward

            for t in xrange(1, T):

                # in most cases, start = 0, end = L
                start = max(0, L - 2 * (T - t))
                end = min(2 * t + 2, L)

                for s in xrange(start, L): # iterate at each position at l'
                    l = (s - 1) / 2 # position in original target label

                    if s % 2 == 0:  # if s(u) is even, it must be blank
                        if s == 0:
                            alphas[s, t] = alphas[s, t - 1] * probs[self.blank, t]
                        else:
                            alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * probs[self.blank, t]
                    elif s == 1 or seq[l] == seq[l - 1]:  # same label twice
                        alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * probs[seq[l], t]
                    else: # not same label
                        alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) * probs[seq[l], t]

                # normalize at current time (prevent underflow)
                c = np.sum(alphas[start:end, t])
                alphas[start:end, t] = alphas[start:end, t] / c
                ll_forward += np.log(c)

            # print "ll_forward: ", ll_forward

            # add to the list
            alphases.append(alphas)
            ll_forwards.append(ll_forward)

            sum = sum - ll_forward


        self.alphases = alphases
        self.ll_forwards = ll_forwards

        return torch.FloatTensor([sum])

    def backward(self, grad_output):
        """
        Returns the gradient of the loss with respect to the input
        """

        input = self.input_np
        seqs = self.seqs_np
        alphases = self.alphases
        ll_forwards = self.ll_forwards

        grads = []
        betases = []
        for i in range(0, input.shape[0]):
            # get data for each sample
            alphas = alphases[i]
            ll_forward = ll_forwards[i]
            seq = seqs[i]
            probs = input[i]

            seq_len = seq.shape[0]  # Length of label sequence (# phones)
            L = 2 * seq_len + 1  # Length of label sequence with blanks
            T = probs.shape[1]  # length of input sequence (time) (m)

            # initialize betas and backwards pass
            betas = np.zeros((L, T))

            betas[-1, -1] = probs[self.blank, -1]
            betas[-2, -1] = probs[seq[-1], -1]
            c = np.sum(betas[:, -1])
            betas[:, -1] = betas[:, -1] / c
            ll_backward = np.log(c)

            for t in xrange(T - 2, -1, -1):
                start = max(0, L - 2 * (T - t))
                end = min(2 * t + 2, L)

                for s in xrange(end - 1, -1, -1):
                    l = (s - 1) / 2

                    if s % 2 == 0:  # blank
                        if s == L - 1:
                            betas[s, t] = betas[s, t + 1] * probs[self.blank, t]
                        else:
                            betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1]) * probs[self.blank, t]
                    elif s == L - 2 or seq[l] == seq[l + 1]:  # same label twice
                        betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1]) * probs[seq[l], t]
                    else:
                        betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1] + betas[s + 2, t + 1]) * probs[seq[l], t]

                # normalize at current time (prevent underflow)
                c = np.sum(betas[start:end, t])
                betas[start:end, t] = betas[start:end, t] / c
                ll_backward += np.log(c)

            betases.append(betas)

            # compute gradient of the loss function with respect to unnormalized input parameters
            grad = np.zeros(probs.shape)
            ab = alphas * betas
            for s in xrange(L): #
                if s % 2 == 0:  # blank
                    grad[self.blank, :] += ab[s, :]
                    ab[s, :] = ab[s, :] / probs[self.blank, :]
                else:
                    grad[seq[(s - 1) / 2], :] += ab[s, :]
                    ab[s, :] = ab[s, :] / (probs[seq[(s - 1) / 2], :])

            absum = np.sum(ab, axis=0)

            # check for underflow or zeros in denominator of the gradient
            llDiff = np.abs(ll_forward - ll_backward)
            if llDiff > 1e-6 or np.sum(absum == 0) > 0:
                print "There is diff in forward/backward LL : %f" % llDiff
                print "Zeros found : (%d/%d)" % (np.sum(absum == 0), absum.shape[0])
                torch.FloatTensor(grads).cuda() if self.cuda else torch.FloatTensor(grads), None

            grad = probs - grad / (probs * absum)

            # add to list
            grads.append(grad)

        return torch.FloatTensor(grads).cuda() if self.cuda else torch.FloatTensor(grads), None