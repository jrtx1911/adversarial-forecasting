import numpy as np
import pytest
from scipy import special
from hmmlearn.base import _BaseHMM
from hmmlearn import hmm
import random



# frameprob = np.asarray([[0.9, 0.2],[0.9, 0.2],[0.1, 0.8],[0.9, 0.2],[0.9, 0.2]])
# model = hmm.GaussianHMM(n_components=2, covariance_type="full")
# model.fit(frameprob)
# model._do_forward_pass(frameprob)
#
#
# Z = _BaseHMM._do_forward_pass(model,frameprob)
# print(Z)

print("HI")



# model1 = hmm.GaussianHMM(n_components=2, covariance_type="full")
# model1.startprob_ = np.array([0.99, 0.01]) # pi. 99% chance functioning, 1% malfunctioning
# model1.transmat_ = np.array([[0.90, 0.10], # A
#                             [0.05, 0.95]])
# model1.means_ = np.array([[0.15, 0.8, 0.05], [0.3, 0.2, 0.5]]) #TODO is this B???
#
#
# model2 = hmm.GaussianHMM(n_components=2, covariance_type="full")
# model2.startprob_ = np.array([0.50, 0.50]) # pi
# model2.transmat_ = np.array([[0.75, 0.25], # A
#                             [0.40, 0.60]])
# model2.means_ = np.array([[0.2, 0.2, 0.6], [0.1, 0.7, 0.2]]) #TODO is this B???
#
#
# model3 = hmm.GaussianHMM(n_components=2, covariance_type="full")
# model3.startprob_ = np.array([0.75, 0.25]) # pi
# model3.transmat_ = np.array([[0.50, 0.50], # A
#                             [0.60, 0.40]])
# model3.means_ = np.array([[0.10, 0.05, 0.85], [0.35, 0.35, 0.30]]) #TODO is this B???


import hmmalgo4 as myhmm

import numpy as np
from hmmlearn import hmm


class HMM(hmm.MultinomialHMM):

    def alpha(self, X):

        alphas = self._do_forward_pass(
            self._compute_log_likelihood(X) )[1]

        return alphas

    def beta(self, X):

        betas = self._do_backward_pass(
            self._compute_log_likelihood(X) )

            # probability of observing this data given the parameters
            # Log P(X| theta)
            # Look at problem 1 in Rabiner paper

        return betas


if __name__ == "__main__":


    transition = np.array(((0.90, 0.10), (0.05, 0.95)))
    emission = np.array(((0.15, 0.8, 0.05), (0.3, 0.2, 0.5)))
    priors = np.array((0.99, 0.01))

    m = HMM(2)

    m.startprob_ = priors
    m.transmat_ = transition
    m.emissionprob_ = emission



    X = np.atleast_2d([0,1,2,2,1]).T

    # print(m.alpha(X))
    # print(m._compute_log_likelihood(X))

################################################################################


# transition = np.array(((0.5, 0.5), (0.3, 0.7)))
# emission = np.array(((0.8, 0.2), (0.4, 0.6)))
# initial = np.array((0.375, 0.625))
# X1 = np.array((0,0,1))



transition = np.array(((0.90, 0.10), (0.05, 0.95)))
emission = np.array(((0.15, 0.8, 0.05), (0.3, 0.2, 0.5)))
initial = np.array((0.99, 0.01))
X1 = np.array((0,1,2,2,1))

def forward(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]

    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]

    return alpha

def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))

    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))

    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])

    return beta




alpha1 = myhmm.forwardPass(X1, transition, emission, initial)
# alpha2 = forward(X1, transition, emission, initial)

print(alpha1)
# print(alpha2)






# P = np.array(((1,2,1),(2,3,1),(3,3,1),(1,1,0),(2,2,0),(3,3,0)))
#
# X1 = np.array((1,2,3,1,2))
#
# Y1 = np.array((2,3,3,1,2))
#
# A1 = np.array((1,1,1,0,0))
#
# X = np.array((1,2,3,3,2))
# Y = np.array((1,1,2,2,3))
# A = np.array((1,1,1,1,1))
#
# T = len(X)
