import numpy as np
import math
import collections

NEG_INF = -float("inf")

class Decoder():
    def decode_best_path(self, input):
        """
        Computes best path given sequence of probability distributions per frame.
        Simply chooses most likely label at each timestep then collapses result to
        remove blanks and repeats.

        Args:
            input( array D = (batch_size, seq_len, classes)): An array of probabilities with dimenssion

        Returns:
            A list of predictions
        """

        hyps = []
        for i in range(0, input.shape[0]):
            probs = input[i]
            best_path = np.argmax(probs, axis=0).tolist()

            # print best_path

            # collapse string
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

        Args:
            probs (array D = (seq_len, clases)): The output probabilities (e.g. post-softmax) for each time step

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

    def edit_distance(self, refs, hyps):
        """
        Edit distance between two sequences reference (ref) and hypothesis (hyp).

        Returns edit distance, number of insertions, deletions and substitutions to
        transform hyp to ref, and number of correct matches.
        """

        dists = []
        inses = []
        delses = []
        subses = []
        corrses = []

        for i in range(0, input.shape[0]):
            ref = refs[i]
            hyp = hyps[i]

            n = len(ref)
            m = len(hyp)

            ins = dels = subs = corrs = 0

            D = np.zeros((n + 1, m + 1))

            D[:, 0] = np.arange(n + 1)
            D[0, :] = np.arange(m + 1)

            for i in xrange(1, n + 1):
                for j in xrange(1, m + 1):
                    if ref[i - 1] == hyp[j - 1]:
                        D[i, j] = D[i - 1, j - 1]
                    else:
                        D[i, j] = min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]) + 1

            i = n
            j = m
            while i > 0 and j > 0:
                if ref[i - 1] == hyp[j - 1]:
                    corrs += 1
                elif D[i - 1, j] == D[i, j] - 1:
                    ins += 1
                    j += 1
                elif D[i, j - 1] == D[i, j] - 1:
                    dels += 1
                    i += 1
                elif D[i - 1, j - 1] == D[i, j] - 1:
                    subs += 1
                i -= 1
                j -= 1

            ins += i
            dels += j

            dists.append(D[-1, -1])
            inses.append(ins)
            delses.append(dels)
            subses.append(subs)
            corrses.append(corrs)

        return dists, inses, delses, subses, corrses

    # def display_edit_diff(self, ref, hyp):
    #     dist, ins, dels, subs, corr = self.edit_distance(ref, hyp)
    #     print "Reference : %s, Hypothesis : %s" % (str(ref), str(hyp))
    #     print "Distance : %d" % dist
    #     print "Ins : %d, Dels : %d, Subs : %d, Corr : %d" % (ins, dels, subs, corr)