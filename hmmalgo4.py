# Created by JCR

import numpy as np
#import random
import helpers
import time
from hmmlearn import hmm # MUST USE hmmlearn version 0.2.6

# TODO:
# use anaconda to add notes/documentation
# Is n_components the dimensionality of A? in hmmlearn
# What is covariance_type? in hmmlearn
# Update check_attack_validity() to work with different probabilities
# Modify forward/backward algorithm to stop computation at timeOfInterest
# Don't need to do forward/backward pass algos for all n. It does it already
# Create class(es) to contain methods. Improve project structure
# Try with longer T=10
# Check attacker parameters, make sure they match paper
# Maybe use numpy arrays instead of lists to improve performance
# Don't use positional arguments, use argument=arg style
# Reap Mark Stamps Paper section 6 for using logs. Products become summations


# QUESTIONS:
# Where to implement summation instead of multiplcation when using logs in alpha/beta

# NOTES:
# Looking for alpha and beta of [timeOfInterest][n]




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
                # print("match",(x[t],y[t]+1,a[t],P[p]))

            # else:
            #     print(x[t],y[t]+1,a[t],P[p])

        if match == 0:
            return 0

        match = 0

    return 1


# Old version that doesnt use hmmlearn library. Also non functional. Error is in the use of N_list, u_list, and u_list_sum
def algo4(X, K, upperLambda, lowerLambda, P, timeOfInterest, c):
    N = lowerLambda[0].shape[0] # number of possible unobservable states
    T = len(X) # observation length
    P_list = [[0 for i in range(K)] for j in range(N)]
    pHat = []
    A = pow(2, T) # number of possible attack variations
    N_list = np.zeros(N)
    Y = pow(3, T) # number of possible observation variations
    u_list = []
    u_list_sum = []



    for k in range(K):

        for n in range(N):
            # print("\nk: ",k,", n: ",n)
            #TODO: the model used here should be one of the attackers estimates. not his own model
            alpha = forwardPass(X, upperLambda[k][0], upperLambda[k][1], upperLambda[k][2])
            beta = backwardPass(X, upperLambda[k][0], upperLambda[k][1])
            P_list[n][k] = alpha[timeOfInterest][n] * beta[timeOfInterest][n]

            # print("\nalpha\n",alpha)
            # print("beta\n",beta)



    for n in range (N):
        sum = 0

        for k in range(K):
            sum = sum + P_list[n][k]
            # print(P_list[n][k])

        pHat.append(sum / K)

        counter = 0
    for a in range(A): # all possible attack/not attack variations. 32 possible (00000, 00001, 00010...)

        a_list = create_seq(a, 2, T)
        cost = compute_cost(c, T, a_list)

        for n in range(N):
            for y in range(Y):
                y_list = create_seq(y, 3, T) #TODO: 2nd parameter should be number rows of emission matrix
                attack_valid = check_attack_validity(P, X, y_list, a_list, T)

                if attack_valid:
                    tempAlpha = forwardPass(y_list, lowerLambda[0], lowerLambda[1], lowerLambda[2])
                    tempBeta = backwardPass(y_list, lowerLambda[0], lowerLambda[1])

                    N_list[n] += (tempAlpha[timeOfInterest][n] * tempBeta[timeOfInterest][n])# * random.random()
                    # print(tempAlpha[timeOfInterest][n] , tempBeta[timeOfInterest][n])
                    # print("\nalpha\n",tempAlpha)
                    # print("\nbeta\n",tempBeta)

            u_list.append((N_list[n] - pHat[n]) * (N_list[n] - pHat[n]))


        sum = 0
        for n in range(N):
            sum += u_list[n]

        sum -= cost

        u_list_sum.append(sum)

        # Find attack with highest utility
    max = u_list_sum[0]
    max_index = 0
    for a in range(A):
        print(u_list_sum[a])
        if u_list_sum[a] > max:
            max = u_list_sum[a]
            max_index = a

    return max_index



def algo4hmm(timeOfInterest, c):
    N = lowerLambda[0].shape[0] # number of possible unobservable states
    K = len(models) - 1 # the number of attacker model estimates
    T = len(X) # observation length
    M = models[0].emissionprob_.shape[1] # set of possible observations
    A = pow(2, T) # number of possible attack variations
    Y = pow(M, T) # number of possible observation variations

     # TODO: need better names for these
    P_list = []
    pHat = []
    N_list = np.zeros(N)
    utility = np.zeros(N) # the temp utility for each n for a given attack
    utilities = np.zeros(A) # the final utilities for each attack, summed over N



    gammas = [] # the alpha * beta value for each n and each y. TODO: is gamma the correct name?


    # find alpha beta for each attacker model estimate and for each state.
    # TODO: is create_seq more time consuming than alpha_pass and beta_pass? also consider space vs time complexity
    for k in range(K):

        alpha = models[k+1].alpha_pass(X)
        beta = models[k+1].beta_pass(X)
        temp_p = []

        for n in range(N):
            temp_p.append(alpha[timeOfInterest][n] * beta[timeOfInterest][n])

        P_list.append(temp_p)



    # find the average for each n over the two attacker model estimates
    for n in range (N):
        sum = 0

        for k in range(K):
            sum +=  P_list[k][n]
            # print(P_list[n][k])

        pHat.append(sum / K)


    # dont need to find alpha/beta for all A so moved it outside loop
    for y in range(Y):
        y_list = create_seq(y, M, T)
        y_list = np.atleast_2d(y_list).T
        alpha = models[0].alpha_pass(y_list)
        beta = models[0].beta_pass(y_list)

        temp_gammas = []

        for n in range(N):
            temp_gammas.append(alpha[timeOfInterest][n] * beta[timeOfInterest][n])

        gammas.append(temp_gammas)


    counter = 0

    for a in range(A): # all possible attack/not attack variations. 32 possible (00000, 00001, 00010...)

        a_list = create_seq(a, 2, T)
        cost = compute_cost(c, T, a_list)

        # calculate n(n)
        for n in range(N):
            N_list[n] = 0
            for y in range(Y):
                # TODO: consider moving these two lines to y loop above and saving results of check_attack_validity for each y. Reduces calls to create_seq()
                y_list = create_seq(y, M, T)
                attack_valid = check_attack_validity(attack_outcomes, X, y_list, a_list, T)

                if attack_valid:
                    N_list[n] += gammas[y][n] #TODO: multiply by probability of attack success. need to update structure of P

            utility[n] = ((N_list[n] - pHat[n]) * (N_list[n] - pHat[n]))


        # sum the values over N to get the average for this attack. must be done after calculating each n(n)
        utilities[a] = utility.sum() - cost



    # just printing all attack utilities here
    print("\na    Utility")
    print("------------------------")
    for idx, util in enumerate(utilities):
        print(idx, ": ",util)


    return utilities.argmax(), utilities



if __name__ == "__main__":
    # X = np.array((0,1,2,2,1))
    # X = np.atleast_2d([0,1,2,2,1]).T
    X = np.atleast_2d([0,1,2,2,1,1,0,2,2,0]).T

    attack_outcomes = np.array(((0,1,1),(1,2,1),(2,2,1),(0,0,0),(1,1,0),(2,2,0)))

########
# Model with 2 states

    transition = np.array(((0.90, 0.10), (0.05, 0.95)))
    emission = np.array(((0.15, 0.8, 0.05), (0.3, 0.2, 0.5)))
    initial = np.array((0.99, 0.01))

    transition1 = np.array(((0.9, 0.1), (0.05, 0.95)))
    emission1 = np.array(((0.15, 0.8, 0.05), (0.3, 0.2, 0.5)))
    initial1 = np.array((0.99, 0.01))

    transition2 = np.array(((0.8, 0.2), (0.1, 0.9)))
    emission2 = np.array(((0.1, 0.7, 0.2), (0.2, 0.3, 0.5)))
    initial2 = np.array((0.99, 0.1))

    lowerLambda = [transition, emission, initial]

    estimate1 = [transition1, emission1, initial1]
    estimate2 = [transition2, emission2, initial2]
    upperLambda = [estimate1, estimate2]

##########
# Model with 3 states

    transition_new = np.array(((0.8, 0.15, 0.05), (0.7, 0.2, 0.1), (0.05, 0.1, 0.85)))
    emission_new = np.array(((0.15, 0.8, 0.05), (0.2, 0.5, 0.3),(0.3, 0.2, 0.5)))
    initial_new = np.array((0.95, 0.04, 0.01))

    transition1_new = np.array(((0.6, 0.25, 0.15), (0.5, 0.3, 0.2), (0.01, 0.04, 0.95)))
    emission1_new = np.array(((0.3, 0.6, 0.1), (0.1, 0.4, 0.5), (0.03, 0.01, 0.96)))
    initial1_new = np.array((0.98, 0.01, 0.01))

    transition2_new = np.array(((0.8, 0.15, 0.05), (0.7, 0.2, 0.1), (0.05, 0.1, 0.85)))
    emission2_new = np.array(((0.15, 0.8, 0.05), (0.1, 0.6, 0.3), (0.05, 0.02, 0.93)))
    initial2_new = np.array((0.8, 0.15, 0.05))

    transition3_new = np.array(((0.95, 0.04, 0.01), (0.9, 0.08, 0.02), (0.5, 0.4, 0.1)))
    emission3_new = np.array(((0.04, 0.95, 0.01), (0.1, 0.75, 0.15), (0.3, 0.2, 0.5)))
    initial3_new = np.array((0.7, 0.2, 0.1))

    lowerLambda_new = [transition_new, emission_new, initial_new]

    estimate1_new = [transition1_new, emission1_new, initial1_new]
    estimate2_new = [transition2_new, emission2_new, initial2_new]
    upperLambda_new = [estimate1_new, estimate2_new]


############

# Using hmmlearn library and the 2 state model from above

    model = HMM(2)
    model.transmat_ = transition
    model.emissionprob_ = emission
    model.startprob_ = initial

    model1 = HMM(2)
    model1.transmat_ = transition1
    model1.emissionprob_ = emission1
    model1.startprob_ = initial1

    model2 = HMM(2)
    model2.transmat_ = transition2
    model2.emissionprob_ = emission2
    model2.startprob_ = initial2

    models = [model, model1, model2]

############



    start = time.time()
    # parameters:
    # X, number of model estimates, all attacker model estimates, attacker model, P, time of interest, cost
    # a_star = algo4(X, 2, upperLambda_new, lowerLambda_new, attack_outcomes, 2, 0.0000000005)
    a_star, utils = algo4hmm(2, 0.0000000005)
    print("UNIQUES: ",len(np.unique(utils)))

    elapsed = time.time()-start

    print("\noptimal attack: ", a_star, ": ", create_seq(a_star, 5, len(X)))
    print("\noptimal attack utility: ", utils[a_star])
    print("\nexecution time: ", elapsed, "seconds\n")

    helpers.save_to_file(a_star, elapsed, utils, X)
