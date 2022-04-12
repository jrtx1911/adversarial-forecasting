# Created by JCR

import numpy as np
#import random
import helpers
import time
from hmmlearn import hmm # MUST USE hmmlearn version 0.2.6

# TODO:
# use anaconda to add notes/documentation
# What is covariance_type? in hmmlearn
# Update check_attack_validity() to work with different probabilities
# Modify forward/backward algorithm to stop computation at timeOfInterest
# Create class(es) to contain methods. Improve project structure
# Check attacker parameters, make sure they match paper
# Maybe use numpy arrays instead of lists to improve performance
# Don't use positional arguments, use argument=arg style


# QUESTIONS:
# Only one model now? with dirichlet distributions based off it?
# For i in {0,1}... am i doing this right?
# Dont need to go over all A and all Y to compute first term mentioned in email? Same in all calculations?
# Constantly changing model parameters for different calculations. Is there any kind of fit needed between changes?
# Where are we confirming attack validity?
# Confirm addition vs multiplication in digammas calculation
# alpha + beta - phat?
# should compute sum over m be outside M loop?

# NOTES:
# ARIMA packages and refresh on idea




class HMM(hmm.MultinomialHMM):

    def __init__(self, n_components):

        super().__init__(n_components)

    def alpha_pass(self, X):

        alphas = self._do_forward_pass(
            self._compute_log_likelihood(X) )[1]

        return alphas

    def beta_pass(self, X):

        betas = self._do_backward_pass(
            self._compute_log_likelihood(X) )

            # probability of observing this data given the parameters
            # Log P(X| theta)
            # Look at problem 1 in Rabiner paper

        return betas



#taken from http://www.adeveloperdiary.com/data-science/machine-learning/forward-and-backward-algorithm-in-hidden-markov-model/
def forwardPass(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]

    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]

    return alpha



def backwardPass(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))

    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))

    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])

    return beta




def create_seq(y, numStatesOfY, T):
    temp = helpers.convert_number_system(y, 10, numStatesOfY)
    templength = len(temp)
    Y = np.zeros(T, dtype=np.int64) #TODO: May need to increase num bits if Y is large
    i = T-templength
    for c in temp:
        Y[i] = ord(c)-48
        i += 1


    return Y



def compute_cost(c, obsLen, currentA):
    cost = 0
    for i in range(obsLen):
        # cost += ord(currentA[i])-48
        cost += currentA[i]

    return c * cost



def check_attack_validity(P, x, y, a, T):
    prows, pcols = P.shape
    # print(x[3])

    match = 0

    for t in range(T):
        for p in range(prows):
            if ((x[t] == P[p][0]) and (y[t] == P[p][1]) and (a[t] == P[p][2])):
                match = 1

        if match == 0:
            return 0

        match = 0

    return 1






def algo4hmm(timeOfInterest, c, S, M):
    # Trying to match notation of Mark Stamp's paper and Tahir Ekin's batch paper
    N = transition.shape[0] # number of possible unobservable states
    # K = len(models) - 1 # the number of attacker model estimates
    T = len(X) # observation length
    M = emission[0].shape[0] # number of possible observations
    A = pow(2, T) # number of possible attack variations
    Y = pow(M, T) # number of possible observation vectors

    # TODO: need better names for these
    Y_list = [] # holds all possible observation vectors (all the y_lists)
    P_list = []
    P_hat = np.zeros(N)
    N_list = np.zeros(N)
    utility = np.zeros(N) # the temp utility for each n for a given attack
    utilities = np.zeros(A) # the final utilities for each attack, summed over N
    utilities_hat = np.zeros(A)
    digammas = np.zeros((Y, N))# the alpha * or + beta value for each n and each y.

    # create Dirichlet distrubtions of model paramters
    P_s = np.zeros((S, N))

    # For S number Dirichlet samples of model
    for s in range(S):
        # For each row of A, B

        temp_initial = np.random.dirichlet(initial_copy, 1)
        np.copyto(model_new.startprob_, temp_initial)

        for n in range(N):
            temp_transition = np.random.dirichlet(transition_copy[n], 1)[0]
            np.copyto(model_new.transmat_[n], temp_transition)
            temp_emission = np.random.dirichlet(emission_copy[n], 1)[0]
            np.copyto(model_new.emissionprob_[n], temp_emission)


        alpha = model_new.alpha_pass(X)
        beta = model_new.beta_pass(X)

        for n in range(N):
            P_s[s][n] = alpha[timeOfInterest][n] + beta[timeOfInterest][n]

    sums = P_s.sum(axis=0)

    for n in range(N):
        P_hat[n] = sums[n] / S





    # Reset model to hold initial values, not a sample from dirichlet distribution
    np.copyto(model_new.transmat_, transition_copy)
    np.copyto(model_new.emissionprob_, emission_copy)
    np.copyto(model_new.startprob_, initial_copy)


    for y in range(Y):
        y_list = create_seq(y, M, T)
        y_list = np.atleast_2d(y_list).T
        Y_list.append(y_list)
        alpha = model_new.alpha_pass(y_list)
        beta = model_new.beta_pass(y_list)


        for n in range(N):
            digammas[y][n] = alpha[timeOfInterest][n] + beta[timeOfInterest][n]



    alpha = model_new.alpha_pass(X)
    beta = model_new.beta_pass(X)
    P_A = 0.0
    for n in range(N):
        P_A += ((alpha[timeOfInterest][n] + beta[timeOfInterest][n] - P_hat[n]) ** 2)


    for a in range(1,A): # all possible attack/not attack variations. 32 possible (00000, 00001, 00010...)
        print("a: ", a)

        a_list = create_seq(a, 2, T)
        cost = compute_cost(c, T, a_list)

        for i in range(1):
            if i == 0:
                utilities[a] = P_A

            if i == 1:
                sum = 0.0
                for n in range(N):
                    for y in range(Y):
                        sum += ((digammas[y][n] - P_hat[n]) ** 2)


        draw = np.random.binomial(1, 0.1, M)

        for m in range(M):
            utilities_hat[a] += utilities[a] - cost

        utilities_hat[0] = P_A

        # just printing all attack utilities here
        print("\na    Utility")
        print("------------------------")
        for idx, util in enumerate(utilities_hat):
            print(idx, ": ",util)


        return utilities_hat.argmax(), utilities_hat





################################################################################





if __name__ == "__main__":
    # X = np.array((0,1,2,2,1))
    X = np.atleast_2d([0,1,2,2,1]).T
    # X = np.atleast_2d([0,1,2,2,1,1,0,2,2,0]).T

    attack_outcomes = np.array(((0,1,1),(1,2,1),(2,2,1),(0,0,0),(1,1,0),(2,2,0)))



################################################################################
# Model with 3 states

    transition = np.array(((0.8, 0.15, 0.05), (0.7, 0.2, 0.1), (0.05, 0.1, 0.85)))
    emission = np.array(((0.15, 0.8, 0.05), (0.2, 0.5, 0.3),(0.3, 0.2, 0.5)))
    initial = np.array((0.95, 0.04, 0.01))

    transition_copy = transition.copy()
    emission_copy = emission.copy()
    initial_copy = initial.copy()






    # transition1_new = np.array(((0.6, 0.25, 0.15), (0.5, 0.3, 0.2), (0.01, 0.04, 0.95)))
    # emission1_new = np.array(((0.3, 0.6, 0.1), (0.1, 0.4, 0.5), (0.03, 0.01, 0.96)))
    # initial1_new = np.array((0.98, 0.01, 0.01))
    #
    # transition2_new = np.array(((0.8, 0.15, 0.05), (0.7, 0.2, 0.1), (0.05, 0.1, 0.85)))
    # emission2_new = np.array(((0.15, 0.8, 0.05), (0.1, 0.6, 0.3), (0.05, 0.02, 0.93)))
    # initial2_new = np.array((0.8, 0.15, 0.05))
    #
    # transition3_new = np.array(((0.95, 0.04, 0.01), (0.9, 0.08, 0.02), (0.5, 0.4, 0.1)))
    # emission3_new = np.array(((0.04, 0.95, 0.01), (0.1, 0.75, 0.15), (0.3, 0.2, 0.5)))
    # initial3_new = np.array((0.7, 0.2, 0.1))

    # lowerLambda_new = [transition_new, emission_new, initial_new]
    #
    # estimate1_new = [transition1_new, emission1_new, initial1_new]
    # estimate2_new = [transition2_new, emission2_new, initial2_new]
    # upperLambda_new = [estimate1_new, estimate2_new]


################################################################################


    # Using hmmlearn library and the 3 state model from above

    model_new = HMM(3)
    model_new.transmat_ = transition
    model_new.emissionprob_ = emission
    model_new.startprob_ = initial

    # model1_new = HMM(3)
    # model1_new.transmat_ = transition1_new
    # model1_new.emissionprob_ = emission1_new
    # model1_new.startprob_ = initial1_new
    #
    # model2_new = HMM(3)
    # model2_new.transmat_ = transition2_new
    # model2_new.emissionprob_ = emission2_new
    # model2_new.startprob_ = initial2_new
    #
    # model3_new = HMM(3)
    # model3_new.transmat_ = transition3_new
    # model3_new.emissionprob_ = emission3_new
    # model3_new.startprob_ = initial3_new

    # models = [model_new, model1_new, model2_new, model3_new]


############


    start = time.time()
    # parameters:
    # X, number of model estimates, all attacker model estimates, attacker model, P, time of interest, cost
    # a_star = algo4(X, 2, upperLambda_new, lowerLambda_new, attack_outcomes, 2, 0.0000000005)
    a_star, utils = algo4hmm(2, 1.0, 1000, 1000000)
    print("UNIQUES: ",len(np.unique(utils)))

    elapsed = time.time()-start

    print("\noptimal attack: ", a_star, ": ", create_seq(a_star, 2, len(X)))
    print("\noptimal attack utility: ", utils[a_star])
    print("\nexecution time: ", elapsed, "seconds\n")

    helpers.save_to_file(a_star, elapsed, utils, X)
