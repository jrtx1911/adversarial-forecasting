
import numpy as np
import pytest
from scipy import special
from hmmlearn.base import _BaseHMM
from hmmlearn import hmm
import random

frameprob = np.asarray([[0.9, 0.2],[0.9, 0.2],[0.1, 0.8],[0.9, 0.2],[0.9, 0.2]])
model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.fit(frameprob)
model._do_forward_pass(frameprob)


Z = _BaseHMM._do_forward_pass(model,frameprob)
print(Z)

#complexity n^2 * T






#####



# from hmmlearn.test_base import TestMonitor
#
# np.random.seed(42)
# model = hmm.GaussianHMM(n_components=3, covariance_type="full")
# model.startprob_ = np.array([0.6, 0.3, 0.1])
# model.transmat_ = np.array([[0.7, 0.2, 0.1],
#                             [0.3, 0.5, 0.2],
#                             [0.3, 0.3, 0.4]])
# model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
# model.covars_ = np.tile(np.identity(2), (3, 1, 1))
# X, Z = model.sample(100)
#
#
#
# temp = _BaseHMM()
# print(dir(_BaseHMM))
# # test = temp._fit_scaling()
# #temp._score_log(X)
#
# frameprob = np.asarray([[0.9, 0.2],[0.9, 0.2],[0.1, 0.8],[0.9, 0.2],[0.9, 0.2]])
#
# log_frameprob = np.log(frameprob)
#
# a,b,c = temp._do_forward_pass(frameprob)
#
# #TestMonitor.test_do_forward_pass()
#
#
# # hmm._do_forward_log_pass(log_frameprob)




# import numpy as np
# import pytest
# from scipy import special
# from hmmlearn.base import _BaseHMM, ConvergenceMonitor
#
#
# class StubHMM(_BaseHMM):
#     """An HMM with hardcoded observation probabilities."""
#     def _compute_log_likelihood(self, X):
#         return self.log_frameprob
#
# class TestBaseAgainstWikipedia:
#     def __init__(self):
#         stuff = 1
#
#     def setup_method(self):
#         # Example from http://en.wikipedia.org/wiki/Forward-backward_algorithm
#         self.frameprob = np.asarray([[0.9, 0.2],
#                                      [0.9, 0.2],
#                                      [0.1, 0.8],
#                                      [0.9, 0.2],
#                                      [0.9, 0.2]])
#         self.log_frameprob = np.log(self.frameprob)
#         h = StubHMM(2)
#         h.transmat_ = [[0.7, 0.3], [0.3, 0.7]]
#         h.startprob_ = [0.5, 0.5]
#         h.log_frameprob = self.log_frameprob
#         h.frameprob = self.frameprob
#         self.hmm = h
#
#     def test_do_forward_pass(self):
#         log_prob, fwdlattice = \
#             self.hmm._do_forward_log_pass(self.log_frameprob)
#         ref_log_prob = -3.3725
#         assert round(log_prob, 4) == ref_log_prob
#         reffwdlattice = np.array([[0.4500, 0.1000],
#                                   [0.3105, 0.0410],
#                                   [0.0230, 0.0975],
#                                   [0.0408, 0.0150],
#                                   [0.0298, 0.0046]])
#         assert np.allclose(np.exp(fwdlattice), reffwdlattice, 4)
#         print(fwdlattice)
#
#
# obj = TestBaseAgainstWikipedia()
# obj.setup_method()
# obj.test_do_forward_pass()
