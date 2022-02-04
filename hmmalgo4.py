# Created by JCR

import numpy as np
import random
import helpers
import time

# TODO:
# use anaconda to add notes/documentation
# Is n_components the dimensionality of A? in hmmlearn
# What is covariance_type? in hmmlearn
# Update checkP() to work with different lengths of attack variations
# Modify forward/backward algorithm to stop computation at timeOfInterest
# Find numObservations dynamically. The 3 different observations (T1,T2,T3... T=temp not time)
# Don't need to do forward/backward pass algos for all n. It does it already
# Create class(es) to contain methods. Improve project structure
# Try with longer T=10
# Check attacker parameters, make sure they match paper
# Implement HMM learn library, test forward pass. Compare with Chema's code

# Check forward/backward pass accuracy. Compare to online examples


# QUESTIONS:
# Time vs space complexity trade off when storing alpha/beta for all Y. This could be very large wiwth long observation sequence. Just store alpha of timeOfInterest
# Why are hmm.alpha values negative? How to implement scaling/log techniques


#NOTES:
# IMPLEMENTED 0 BASED INDEXING FOR ALL VARIABLES: X,Y,P
# Looking for alpha and beta of [timeOfInterest][n]


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



def computeCost(c, obsLen, currentA):
    cost = 0
    for i in range(obsLen):
        # cost += ord(currentA[i])-48
        cost += currentA[i]

    return c * cost


def checkP(P, x, y, a, T):
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




def algo4(X, K, upperLambda, lowerLambda, P, timeOfInterest, c):
    A_rows, A_cols = lowerLambda[0].shape
    N = A_rows # used to be D
    T = len(X)
    P_list = [[0 for i in range(K)] for j in range(N)]
    pHat = []
    A = pow(2, T)
    # A_list = createA(T) # deprecated
    # C = [] # deprecated
    N_list = np.zeros(N)
    Y = pow(3, T)
    u_list = []
    u_list_sum = []




    for k in range(K):

        for n in range(N):
            #TODO: the model used here should be one of the attackers estimates. not his own model
            alpha = forwardPass(X, upperLambda[k][0], upperLambda[k][1], upperLambda[k][2])
            beta = backwardPass(X, upperLambda[k][0], upperLambda[k][1])
            P_list[n][k] = alpha[timeOfInterest][n] * beta[timeOfInterest][n]

            # print("\nalpha\n",alpha)
            # print("\nbeta\n",beta)



    for n in range (N):
        sum = 0

        for k in range(K):
            sum = sum + P_list[n][k]
            # print(P_list[n][k])

        pHat.append(sum / K)

        counter = 0
    for a in range(A): # all possible attack/not attack variations. 32 possible (00000, 00001, 00010...)

        a_list = create_seq(a, 2, T)
        cost = computeCost(c, T, a_list)

        for n in range(N):
            for y in range(Y):
                y_list = create_seq(y, 3, T) #TODO: 2nd parameter should be number rows of emission matrix
                p_something = checkP(P, X, y_list, a_list, T)

                if p_something:
                    tempAlpha = forwardPass(y_list, lowerLambda[0], lowerLambda[1], lowerLambda[2])
                    tempBeta = backwardPass(y_list, lowerLambda[0], lowerLambda[1])

                    N_list[n] += (tempAlpha[timeOfInterest][n] * tempBeta[timeOfInterest][n])
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
        print("u[",a,"]: ", u_list_sum[a], file=f)
        if u_list_sum[a] > max:
            max = u_list_sum[a]
            max_index = a

    return max_index




X = np.array((0,1,2,2,1,0,1,2,1,1))
P = np.array(((0,1,1),(1,2,1),(2,2,1),(0,0,0),(1,1,0),(2,2,0)))
# poisoining outcome


transition = np.array(((0.90, 0.10), (0.05, 0.95)))
emission = np.array(((0.15, 0.8, 0.05), (0.3, 0.2, 0.5)))
initial = np.array((0.99, 0.01))

transition1 = np.array(((0.75, 0.25), (0.4, 0.6)))
emission1 = np.array(((0.2, 0.2, 0.6), (0.1, 0.7, 0.2)))
initial1 = np.array((0.5, 0.5))

transition2 = np.array(((0.5, 0.5), (0.6, 0.5)))
emission2 = np.array(((0.1, 0.05, 0.85), (0.35, 0.35, 0.3)))
initial2 = np.array((0.75, 0.25))

lowerLambda = [transition, emission, initial]

estimate1 = [transition1, emission1, initial1]
estimate2 = [transition2, emission2, initial2]
upperLambda = [estimate1, estimate2]

f = open("results.txt", "w")

# X, number of model estimates, all attacker model estimates, attacker model, P, time of interest, cost
start = time.time()
a_star = algo4(X, 2, upperLambda, lowerLambda, P, 4, 0.00000003)
elapsed = time.time()-start

print(a_star)
print("\noptimal attack: ", a_star,file=f)

print(elapsed)
print("\nexecution time: ", elapsed, file=f)



f.close()
