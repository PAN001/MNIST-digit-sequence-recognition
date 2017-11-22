import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from model import *
import random

# deterministic
manual_seed = 1234
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
cuda = False

class CTCLoss(torch.autograd.Function):
    # def __init__(self):
    #     super(CTC_Loss,self).__init__()
    #     self.sequence = None
    #     self.parameters = None
    #     self.batch_size = None
    #     self.llForwardList = None
    #     return
    # @staticmethod
    def forward(self, parameters, sequence, blank=10, is_prob=True):
        self.sequence = sequence.numpy()
        self.parameters = parameters.numpy()
        self.batch_size = self.parameters.shape[0]
        self.llForwardList = []
        self.alphas = []
        self.blank = 10
        loss = 0
        for i in range(self.batch_size):
            params = self.parameters[i]
            seq = self.sequence[i]
            # print("params size:",params.shape)
            seqLen = seq.shape[0]  # Length of label sequence (# phones)
            numphones = params.shape[0]  # Number of labels
            L = 2 * seqLen + 1  # Length of label sequence with blanks
            T = params.shape[1]  # Length of utterance (time)

            alphas = np.zeros((L, T))

            # Keep for gradcheck move this, assume NN outputs probs
            if not is_prob:
                params = params - np.max(params, axis=0)
                params = np.exp(params)
                params = params / np.sum(params, axis=0)

            # print("seq ",seq)
            # print("seq 0",seq[0])
            # print("blank",blank)
            # Initialize alphas and forward pass
            alphas[0, 0] = params[blank, 0]
            alphas[1, 0] = params[seq[0], 0]
            c = np.sum(alphas[:, 0])
            # print("c: ",c)
            alphas[:, 0] = alphas[:, 0] / c
            llForward = np.log(c)
            for t in range(1, T):
                start = max(0, L - 2 * (T - t))
                end = min(2 * t + 2, L)
                for s in range(start, L):
                    l = (s - 1) // 2
                    # blank
                    if s % 2 == 0:
                        if s == 0:
                            alphas[s, t] = alphas[s, t - 1] * params[blank, t]
                        else:
                            alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[blank, t]
                    # same label twice
                    elif s == 1 or seq[l] == seq[l - 1]:
                        alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[seq[l], t]
                    else:
                        alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) * params[
                            seq[l], t]

                # normalize at current time (prevent underflow)
                c = np.sum(alphas[start:end, t])
                alphas[start:end, t] = alphas[start:end, t] / c
                llForward += np.log(c)
                # print("logc:",np.log(c))
            self.llForwardList.append(-llForward)
            self.alphas.append(alphas)
            loss = loss - llForward
            print(self.llForwardList)

        hypList = self.decode_best_path(self.parameters)
        print("")
        print("prediction:\n", hypList)
        print("target label:   ", self.sequence)
        return torch.FloatTensor([loss])

    # @staticmethod
    def backward(self, grad_output, blank=10):
        gradList = []
        for i in range(self.batch_size):
            params = self.parameters[i]
            seq = self.sequence[i]
            llForward = self.llForwardList[i]

            seqLen = seq.shape[0]  # Length of label sequence (# phones)
            numphones = params.shape[0]  # Number of labels
            L = 2 * seqLen + 1  # Length of label sequence with blanks
            T = params.shape[1]  # Length of utterance (time)
            betas = np.zeros((L, T))
            # Initialize betas and backwards pass
            alphas = self.alphas[i]
            betas[-1, -1] = params[blank, -1]
            betas[-2, -1] = params[seq[-1], -1]
            c = np.sum(betas[:, -1])
            betas[:, -1] = betas[:, -1] / c
            llBackward = np.log(c)
            for t in range(T - 2, -1, -1):
                start = max(0, L - 2 * (T - t))
                end = min(2 * t + 2, L)
                for s in range(end - 1, -1, -1):
                    l = (s - 1) // 2
                    # blank
                    if s % 2 == 0:
                        if s == L - 1:
                            betas[s, t] = betas[s, t + 1] * params[blank, t]
                        else:
                            betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1]) * params[blank, t]
                    # same label twice
                    elif s == L - 2 or seq[l] == seq[l + 1]:
                        betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1]) * params[seq[l], t]
                    else:
                        betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1] + betas[s + 2, t + 1]) * params[seq[l], t]

                c = np.sum(betas[start:end, t])
                betas[start:end, t] = betas[start:end, t] / c
                llBackward += np.log(c)

            # Compute gradient with respect to unnormalized input parameters
            grad = np.zeros(params.shape)
            ab = alphas * betas
            for s in range(L):
                # blank
                if s % 2 == 0:
                    grad[blank, :] += ab[s, :]
                    ab[s, :] = ab[s, :] / params[blank, :]
                else:
                    grad[seq[(s - 1) // 2], :] += ab[s, :]
                    ab[s, :] = ab[s, :] / (params[seq[(s - 1) // 2], :])
            absum = np.sum(ab, axis=0)

            # Check for underflow or zeros in denominator of gradient
            llDiff = np.abs(llForward + llBackward)
            # print("llDiff",llDiff)
            # print("forward",llForward)
            # print("backward",llBackward)
            if llDiff > 1e-5 or np.sum(absum == 0) > 0:
                print("Diff in forward/backward LL : %f", llDiff)
                print("Zeros found : (%d/%d)", (np.sum(absum == 0), absum.shape[0]))
                return torch.FloatTensor(gradList), None

            grad = params - grad / (params * absum)
            gradList.append(grad)
        # print(gradList)
        return torch.FloatTensor(gradList), None

    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        input = self.parameters
        seqs = self.sequence
        alphases = self.alphas
        ll_forwards = self.llForwardList

        grads = []
        betases = []
        for i in range(0, self.batch_size):
            # get data for each sample
            alphas = alphases[i]
            ll_forward = ll_forwards[i]
            seq = seqs[i]
            params = input[i]

            seq_len = seq.shape[0]  # Length of label sequence (# phones)
            L = 2 * seq_len + 1  # Length of label sequence with blanks
            T = params.shape[1]  # length of input sequence (time) (m)

            # initialize betas and backwards pass
            betas = np.zeros((L, T))

            betas[-1, -1] = params[self.blank, -1]
            betas[-2, -1] = params[seq[-1], -1]
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
                            betas[s, t] = betas[s, t + 1] * params[self.blank, t]
                        else:
                            betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1]) * params[self.blank, t]
                    elif s == L - 2 or seq[l] == seq[l + 1]:  # same label twice
                        betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1]) * params[seq[l], t]
                    else:
                        betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1] + betas[s + 2, t + 1]) * params[seq[l], t]

                # normalize at current time (prevent underflow)
                c = np.sum(betas[start:end, t])
                betas[start:end, t] = betas[start:end, t] / c
                ll_backward += np.log(c)

            betases.append(betas)

            # compute gradient of the loss function with respect to unnormalized input parameters
            grad = np.zeros(params.shape)
            ab = alphas * betas
            for s in xrange(L): #
                if s % 2 == 0:  # blank
                    grad[self.blank, :] += ab[s, :]
                    ab[s, :] = ab[s, :] / params[self.blank, :]
                else:
                    grad[seq[(s - 1) / 2], :] += ab[s, :]
                    ab[s, :] = ab[s, :] / (params[seq[(s - 1) / 2], :])

            absum = np.sum(ab, axis=0)

            # check for underflow or zeros in denominator of gradient
            llDiff = np.abs(ll_forward + ll_backward)
            if llDiff > 1e-5 or np.sum(absum == 0) > 0:
                print "Diff in forward/backward LL : %f" % llDiff
                print "Zeros found : (%d/%d)" % (np.sum(absum == 0), absum.shape[0])
                # return -ll_forward, grad, True

            grad = params - grad / (params * absum)

            # add to list
            grads.append(grad)

        # print grads
        # print "grads: ", grads
        return torch.FloatTensor(grads).cuda() if cuda else torch.FloatTensor(grads), None

    def decode_best_path(self, probs, ref=None, blank=10):
        hypList = []
        distList = []
        for e in range(self.batch_size):
            best_path = np.argmax(probs[e], axis=0).tolist()

            # Collapse phone string
            hyp = []
            for i, b in enumerate(best_path):
                # ignore blanks
                if b == blank:
                    continue
                # ignore repeats
                elif i != 0 and b == best_path[i - 1]:
                    continue
                else:
                    hyp.append(b)


        return hypList






