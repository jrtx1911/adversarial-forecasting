import numpy as np
import random

# TODO: use anaconda to add notes/documentation
# Is n_components the dimensionality of A? in hmmlearn
# What is covariance_type? in hmmlearn
# Update checkP() to work with different lengths of attack variations
# Modify forward/backward algorithm to stop computation at timeOfInterest
# Find numObservations dynamically. The 3 different observations (T1,T2,T3... T=temp not time)
# Don't need to do forward/backward pass algos for all n. It does it already

# Check forward/backward pass accuracy. Compare to online examples


# QUESTIONS:
# IMPLEMENTED 0 BASED INDEXING FOR ALL VARIABLES: X,Y,P
# Consider moving for A loop. Recalculating of alpha beta unncessary for each attack
# Alpha and beta values constantly repeat. Is this expected? Verify accuracy of forward/backward pass algos
# Is there a forward/backward pass example with numbers?

#NOTES:
# IMPLEMENTED 0 BASED INDEXING FOR ALL VARIABLES: X,Y,P
# Looking for alpha and beta of [timeOfInterest][n]



def convert_number_system(input_number, input_base, output_base):
    '''
    function that calculates numbers from one base to the other
    returns: int, converted number
    '''

    #list that holds the numbers to output in the end
    remainder_list = []

    #start value for sum_base_10. All calculations go thorugh base-10.
    sum_base_10 = 0

    #validate_input


    if output_base == 2:
        binary_repr = bin(input_number)
        return (binary_repr[2:])

    # we coulc use python's built in hex(), but it is more fun not to...
    #if output_base == 16:
        #hex_repr = hex(input_number)
        #return hex_repr[2:]

    # we want to convert to base-10 before the actual calculation:
    elif input_base != 10:

        # reverse the string to start calculating from the least significant number
        reversed_input_number = input_number[::-1]

        #check if user typed in letter outside HEX range.
        hex_helper_dict = {'a' : 10 , 'b' : 11 , 'c' : 12 , 'd' : 13 , 'e' : 14 , 'f' : 15}


        for index, number in enumerate(reversed_input_number):
            for key,value in hex_helper_dict.items():
                if str(number).lower() == key:
                    number = value

            sum_base_10 += (int(number)*(int(input_base)**index))

    # if the number is already in Base-10, we can start the convertion
    elif input_base == 10:
        sum_base_10 = int(input_number)


    # we loop through until we hit 0. When we hit 0, we have our number.
    while sum_base_10 > 0:

        #find number to pass further down the loop
        divided = sum_base_10// int(output_base)

        #find remainder to keep
        remainder_list.append(str(sum_base_10 % int(output_base)))

        # the new value to send to the next iteration
        sum_base_10 = divided


    #fix the list and send a number:
    return_number = ''

    # if the user asked for a Hexadesimal output, we need to convert
    # any number from 10 and up.
    if output_base == 16:
        hex_dict = {10 : 'a' , 11 : 'b' , 12 : 'c' , 13 : 'd' , 14 : 'e' , 15 : 'f'}

        #loop through remainder_list and convert 10+ to letters.
        for index, each in enumerate(remainder_list):
            for key, value in hex_dict.items():
                if each == str(key):
                    remainder_list[index] = value

    #return the number:
    else:
        for each in remainder_list[::-1]:
            return_number += each

        return (return_number)
    #else:
        #return ('invalid input... Please Try Again')




#taken from http://www.adeveloperdiary.com/data-science/machine-learning/forward-and-backward-algorithm-in-hidden-markov-model/
def forwardPass(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]

    for t in range(1, V.shape[0]): # TODO: from 1 to timeOfInterest
        for j in range(a.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]-1]

    return alpha



def backwardPass(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))

    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))

    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1): # TODO: from 5 to 3
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])

    return beta





# obsLen is T. the length of obersvation. DEPRECATED
def createA(obsLen): #only works for 2 states. Counts up in binary
    max = pow(2, obsLen)
    A = [] * obsLen
    for i in range(max):
        temp = str(bin(i))
        temp = temp[2:]
        tempLen = len(temp)
        a = ""

        for j in range(obsLen-tempLen):
            a = a + '0'

        tempLen = len(a)
        for j in range(len(a),obsLen):
            a = a + temp[j-tempLen]

        A.append(a)

    return A


def create_seq(y, numStatesOfY, T):
    temp = convert_number_system(y, 10, numStatesOfY)
    templength = len(temp)
    Y = np.zeros(T, dtype=np.int64)
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
                y_list = create_seq(y, 3, T)
                # print(y_list)

                # forward/backward algos take V, a, b, pi. backward doesnt get pi
                tempAlpha = forwardPass(y_list, lowerLambda[0], lowerLambda[1], lowerLambda[2])
                tempBeta = backwardPass(y_list, lowerLambda[0], lowerLambda[1])
                p_something = checkP(P, X, y_list, a_list, T)

                N_list[n] += (tempAlpha[timeOfInterest][n] * tempBeta[timeOfInterest][n] * p_something)
                # print("\nalpha\n",tempAlpha)
                # print("\nbeta\n",tempBeta)

            u_list.append((N_list[n] - pHat[n]) * (N_list[n] - pHat[n]))


        sum = 0
        for n in range(N):
            sum += u_list[n]

        sum -= cost

        u_list_sum.append(sum)

    max = u_list_sum[0]
    max_index = 0
    for a in range(A):
        print(u_list_sum[a])
        if u_list_sum[a] > max:
            max = u_list_sum[a]
            max_index = a

    return max_index




X = np.array((0,1,2,2,1))
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


# X, number of model estimates, all attacker model estimates, attacker model, P, time of interest, cost
a_star = algo4(X, 2, upperLambda, lowerLambda, P, 2, 0)

print(a_star)




##### TESTING BELOW ######

# Y = 243
#
#
#
# for y in range(Y):
#     y_list = create_y(y, 2, 5)
#     tempY = np.array((3,3,3,3,3))
#     tempY2 = np.array((1,1,1,1,1))
#     tempA = np.array((1,1,1,1,1))
#     tempX = np.array((3,3,3,3,3))
#
#
#     # pass algos take V, a, b, pi. backward doesnt get pi
#     tempAlpha = forwardPass(tempY2, lowerLambda[0], lowerLambda[1], lowerLambda[2])
#     tempBeta = backwardPass(tempY2, lowerLambda[0], lowerLambda[1])
#     p_something = checkP(P, tempX, tempY2, tempA, 5)
#     print(p_something)
#
#
#
#     tempN = tempAlpha[2][0] * tempBeta[2][0] * p_something
#     # print(tempN)
