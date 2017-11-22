# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import numpy as np
dtype = torch.FloatTensor
import torch.nn.functional as F
import math
import collections

NEG_INF = -float("inf")

# 继承torch.autograd.Function，拓展numpy
class CTCLoss(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    def __init__(self, cuda):
        super(CTCLoss, self).__init__()
        self.cuda = cuda

    def forward(self, input, seqs, blank = 10):
        """
        CTC loss function.

        input - D = (batch_size, classes, seq_len): A tensor of network output
        seqs - D = (batch_size, seq_len)
        params - D = (classes, seq_len): matrix of classes-D probability distributions over seq_len frames.
        seq - D = (seq_len): sequence of phone id's for given example.

        Returns objective and gradient.
        """

        self.blank = blank
        self.batch_size = input.shape[0]

        input_np = input.cpu().numpy() if self.cuda else input.numpy()
        seqs_np = seqs.cpu().numpy() if self.cuda else seqs.numpy()

        self.input_np = input_np
        self.seqs_np = seqs_np

        alphases = []
        ll_forwards = []

        sum = 0.0 # the multi

        for i in range(0, input.shape[0]): # iterate over each training sample
            params = input_np[i]
            seq = seqs_np[i]

            seq_len = seq.shape[0]  # length of label sequence (# expected digits)
            L = 2 * seq_len + 1  # length of label sequence with blanks. e.g. abc -> _a_b_c_
            T = params.shape[1]  # length of input sequence (time) (m)

            # forward dynamic table: L * T
            # alphas[u, t] represent the sum of probability of all paths outputing l'u and time t
            alphas = np.zeros((L, T))

            # initialize alphas and forward pass
            alphas[0, 0] = params[self.blank, 0]  # a(u, t)
            alphas[1, 0] = params[seq[0], 0]
            c = np.sum(alphas[:, 0])

            alphas[:, 0] = alphas[:, 0] / c
            ll_forward = np.log(c) # log liklihood of forward

            for t in xrange(1, T):

                # in most cases, start = 0, end = L
                start = max(0, L - 2 * (T - t))
                end = min(2 * t + 2, L)

                for s in xrange(start, L): # iterate at each position at l'
                    l = (s - 1) / 2 # pisition in original target label

                    if s % 2 == 0:  # if s(u) is even, it must be blank
                        if s == 0:
                            alphas[s, t] = alphas[s, t - 1] * params[self.blank, t]
                        else:
                            alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[self.blank, t]
                    elif s == 1 or seq[l] == seq[l - 1]:  # same label twice
                        alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[seq[l], t]
                    else: # not same label
                        alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) * params[seq[l], t]

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
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
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
            llDiff = np.abs(ll_forward - ll_backward)
            if llDiff > 1e-5 or np.sum(absum == 0) > 0:
                print "Diff in forward/backward LL : %f" % llDiff
                print "Zeros found : (%d/%d)" % (np.sum(absum == 0), absum.shape[0])
                # return -ll_forward, grad, True

            grad = params - grad / (params * absum)

            # add to list
            grads.append(grad)

        return torch.FloatTensor(grads).cuda() if self.cuda else torch.FloatTensor(grads), None

    def decode_best_path(self, input):
        """
        Computes best path given sequence of probability distributions per frame.
        Simply chooses most likely label at each timestep then collapses result to
        remove blanks and repeats.

        input: A tensor of probabilities with dimenssion D = (batch_size, seq_len, classes)

        Returns hypothesis transcription
        """

        # Compute best path
        # input = input.data.cpu().numpy() if cuda else input.data.numpy()
        # input = input.transpose(0, 2, 1)  # D = (batch_size, classes, seq_len)
        hyps = []
        for i in range(0, input.shape[0]):
            probs = input[i]
            best_path = np.argmax(probs, axis=0).tolist()

            # print best_path

            # Collapse phone string
            hyp = []
            for i, b in enumerate(best_path):
                # ignore blanks
                if b == self.blank:
                    continue
                # ignore repeats
                elif i != 0 and b == best_path[i - 1]:
                    continue
                else:
                    hyp.append(b)

            hyps.append(hyp)

        return hyps

    def make_new_beam(self):
        fn = lambda: (NEG_INF, NEG_INF)
        return collections.defaultdict(fn)

    def logsumexp(self, *args):
        """
        Stable log sum exp.
        """
        if all(a == NEG_INF for a in args):
            return NEG_INF
        a_max = max(args)
        lsp = math.log(sum(math.exp(a - a_max)
                           for a in args))
        return a_max + lsp

    def decode_beam(self, input, beam_size=100):
        """
        Performs inference for the given output probabilities.

        Arguments:
            probs (D = (seq_len, clases)): The output probabilities (e.g. post-softmax) for each
              time step. Should be an array of shape (time x output dim).

            beam_size (int): Size of the beam to use during inference.

            blank (int): Index of the CTC blank label.
        Returns the output label sequence and the corresponding negative
        log-likelihood estimated by the decoder.
        """

        hyps = []
        scores = []
        for i in range(0, input.shape[0]):
            probs = input[i]

            T, S = probs.shape
            probs = np.log(probs)

            # Elements in the beam are (prefix, (p_blank, p_no_blank))
            # Initialize the beam with the empty sequence, a probability of
            # 1 for ending in blank and zero for ending in non-blank
            # (in log space).
            beam = [(tuple(), (0.0, NEG_INF))]

            for t in range(T):  # iterate over time

                # A default dictionary to store the next step candidates.
                next_beam = self.make_new_beam()

                for s in range(S):  # Loop over vocab
                    p = probs[t, s]

                    # The variables p_b and p_nb are respectively the
                    # probabilities for the prefix given that it ends in a
                    # blank and does not end in a blank at this time step.
                    for prefix, (p_b, p_nb) in beam:  # Loop over beam

                        # If we propose a blank the prefix doesn't change.
                        # Only the probability of ending in blank gets updated.
                        if s == self.blank:
                            n_p_b, n_p_nb = next_beam[prefix]
                            n_p_b = self.logsumexp(n_p_b, p_b + p, p_nb + p)
                            next_beam[prefix] = (n_p_b, n_p_nb)
                            continue

                        # Extend the prefix by the new character s and add it to
                        # the beam. Only the probability of not ending in blank
                        # gets updated.
                        end_t = prefix[-1] if prefix else None
                        n_prefix = prefix + (s,)
                        n_p_b, n_p_nb = next_beam[n_prefix]
                        if s != end_t:
                            n_p_nb = self.logsumexp(n_p_nb, p_b + p, p_nb + p)
                        else:
                            # We don't include the previous probability of not ending
                            # in blank (p_nb) if s is repeated at the end. The CTC
                            # algorithm merges characters not separated by a blank.
                            n_p_nb = self.logsumexp(n_p_nb, p_b + p)

                        # *NB* this would be a good place to include an LM score.
                        next_beam[n_prefix] = (n_p_b, n_p_nb)

                        # If s is repeated at the end we also update the unchanged
                        # prefix. This is the merging case.
                        if s == end_t:
                            n_p_b, n_p_nb = next_beam[prefix]
                            n_p_nb = self.logsumexp(n_p_nb, p_nb + p)
                            next_beam[prefix] = (n_p_b, n_p_nb)

                    # Sort and trim the beam before moving on to the
                    # next time-step.
                    beam = sorted(next_beam.items(),
                                  key=lambda x: self.logsumexp(*x[1]),
                                  reverse=True)
                    beam = beam[:beam_size]

            best = beam[0]
            hyps.append(best[0])
            scores.append(-self.logsumexp(*best[1]))
        return hyps, scores

# # 拓展Module
# class CTCLoss(torch.nn.Module):
#     def forward(self, input, label):
#         """
#         In the forward pass we receive a Tensor containing the input and return a
#         Tensor containing the output. You can cache arbitrary Tensors for use in the
#         backward pass using the save_for_backward method.
#         """
#
#         self.save_for_backward(input)
#         return input.clamp(min=0)
#
#     def backward(self, grad_output):
#         """
#         In the backward pass we receive a Tensor containing the gradient of the loss
#         with respect to the output, and we need to compute the gradient of the loss
#         with respect to the input.
#         """
#         input, = self.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input[input < 0] = 0
#         return grad_input

# (batch_size * seq_len, classes)

# cuda = False
# batch_size = 50
# seq_len = 100
# classes = 11
#
# dataset_data = np.load("./dataset/data.npy")
# dataset_labels = np.load("./dataset/labels.npy")
# dataset_labels = dataset_labels.astype(int)
#
# criterion = CTCLoss()
# out = Variable(torch.rand(batch_size, seq_len, classes).type(dtype), requires_grad=True)
# data = torch.Tensor(dataset_data)
# shape = data.shape
# labels = torch.IntTensor(dataset_labels)
# labels = Variable(labels.cuda()) if cuda else Variable(labels)
#
# loss = criterion(out, labels)
#
# # loss.backward()


# time = 50
# output_dim = 8
#
# probs = np.random.rand(time, output_dim)
# probs = probs / np.sum(probs, axis=1, keepdims=True)
#
# criterion = CTCLoss()
# labels, score = criterion.decode_beam(probs)
# print labels
# print("Score {:.3f}".format(score))