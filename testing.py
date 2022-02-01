import numpy as np




'''
Say a=1,0,0,1,0

d=State F

For all Y====(T1,T1,T1,T2,T3)
Say a=1,0,0,1,0 (1 out of 32)
d==State F or M
State F
For all Y—3^5–243 alternatives
Y=(T1,T1,T1,T1,T1)
Y: modified data.
HMMlearn: alpha, beta for data of Y=(T1,T1,T1,T1,T1) and state F
Using Attacckers own HMM parametrization
P(Y|X,a)
Tahir Ekin to Me (Direct Message) (11:51 AM)
PApYt“T2|Xt“T1,at“1)=1;PApYt“T3|Xt“T2,at“1)=1;PApYt“T3|Xt“T3,at“1)=1;PApYt“T1|Xt“T1,at“0)=1;PApYt“T2|Xt“T2,at“0)=1;PApYt“T3|Xt“T3,at“0)=1 for allt“1,..,T.PApYt“yt|Xt“xt,atqare equal to zero for remaining combinations
X given
X= T1, T2, T3,T3,T2
Tahir Ekin to Me (Direct Message) (11:53 AM)
P(Y1=T1| X1=T1, a=1)
P(Y2=T1 | X2=T2,a=0)
P( Y|a,X)


Say a=1,0,0,1,0
c=1000
dolalrs
dollars
c=250 $
a=1,0,0,1,0
c(a)=1*250+0+0+1*250+0
'''




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





def create_y(y, numStates, T):
    temp = convert_number_system(y, 10, numStates)
    templength = len(temp)
    Y = np.zeros(T, dtype=np.int32)
    i = T-templength
    for c in temp:
        Y[i] = ord(c)-48
        i += 1


    return Y

for i in range(243):
    print(create_y(i,3,5))













def createY(numStates, obsLen):
    Y_list = []
    Y_list.append(np.array([1,1,1,1,1]))
    Y_list.append(np.array([1,1,1,1,2]))
    Y_list.append(np.array([1,1,1,1,3]))
    Y_list.append(np.array([1,1,1,2,1]))
    Y_list.append(np.array([1,1,1,2,2]))
    Y_list.append(np.array([1,1,1,2,3]))
    Y_list.append(np.array([1,1,1,3,1]))
    Y_list.append(np.array([1,1,1,3,2]))
    Y_list.append(np.array([1,1,1,3,3]))
    Y_list.append(np.array([1,1,2,1,1]))
    Y_list.append(np.array([1,1,2,1,2]))
    Y_list.append(np.array([1,1,2,1,3]))
    Y_list.append(np.array([1,1,2,2,1]))
    Y_list.append(np.array([1,1,2,2,2]))
    Y_list.append(np.array([1,1,2,2,3]))
    Y_list.append(np.array([1,1,2,3,1]))
    Y_list.append(np.array([1,1,2,3,2]))
    Y_list.append(np.array([1,1,2,3,3]))
    Y_list.append(np.array([1,1,3,1,1]))
    Y_list.append(np.array([1,1,3,1,2]))
    Y_list.append(np.array([1,1,3,1,3]))
    Y_list.append(np.array([1,1,3,2,1]))
    Y_list.append(np.array([1,1,3,2,2]))
    Y_list.append(np.array([1,1,3,2,3]))
    Y_list.append(np.array([1,1,3,3,1]))
    Y_list.append(np.array([1,1,3,3,2]))
    Y_list.append(np.array([1,1,3,3,3]))
    Y_list.append(np.array([1,2,1,1,1]))
    Y_list.append(np.array([1,2,1,1,2]))
    Y_list.append(np.array([1,2,1,1,3]))
    Y_list.append(np.array([1,2,1,2,1]))
    Y_list.append(np.array([1,2,1,2,2]))
    Y_list.append(np.array([1,2,1,2,3]))
    Y_list.append(np.array([1,2,1,3,1]))
    Y_list.append(np.array([1,2,1,3,2]))
    Y_list.append(np.array([1,2,1,3,3]))
    Y_list.append(np.array([1,2,2,1,1]))
    Y_list.append(np.array([1,2,2,1,2]))
    Y_list.append(np.array([1,2,2,1,3]))
    Y_list.append(np.array([1,2,2,2,1]))
    Y_list.append(np.array([1,2,2,2,2]))
    Y_list.append(np.array([1,2,2,2,3]))
    Y_list.append(np.array([1,2,2,3,1]))
    Y_list.append(np.array([1,2,2,3,2]))
    Y_list.append(np.array([1,2,2,3,3]))
    Y_list.append(np.array([1,2,3,1,1]))
    Y_list.append(np.array([1,2,3,1,2]))
    Y_list.append(np.array([1,2,3,1,3]))
    Y_list.append(np.array([1,2,3,2,1]))
    Y_list.append(np.array([1,2,3,2,2]))
    Y_list.append(np.array([1,2,3,2,3]))
    Y_list.append(np.array([1,2,3,3,1]))
    Y_list.append(np.array([1,2,3,3,2]))
    Y_list.append(np.array([1,2,3,3,3]))
    Y_list.append(np.array([1,3,1,1,1]))
    Y_list.append(np.array([1,3,1,1,2]))
    Y_list.append(np.array([1,3,1,1,3]))
    Y_list.append(np.array([1,3,1,2,1]))
    Y_list.append(np.array([1,3,1,2,2]))
    Y_list.append(np.array([1,3,1,2,3]))
    Y_list.append(np.array([1,3,1,3,1]))
    Y_list.append(np.array([1,3,1,3,2]))
    Y_list.append(np.array([1,3,1,3,3]))
    Y_list.append(np.array([1,3,2,1,1]))
    Y_list.append(np.array([1,3,2,1,2]))
    Y_list.append(np.array([1,3,2,1,3]))
    Y_list.append(np.array([1,3,2,2,1]))
    Y_list.append(np.array([1,3,2,2,2]))
    Y_list.append(np.array([1,3,2,2,3]))
    Y_list.append(np.array([1,3,2,3,1]))
    Y_list.append(np.array([1,3,2,3,2]))
    Y_list.append(np.array([1,3,2,3,3]))
    Y_list.append(np.array([1,3,3,1,1]))
    Y_list.append(np.array([1,3,3,1,2]))
    Y_list.append(np.array([1,3,3,1,3]))
    Y_list.append(np.array([1,3,3,2,1]))
    Y_list.append(np.array([1,3,3,2,2]))
    Y_list.append(np.array([1,3,3,2,3]))
    Y_list.append(np.array([1,3,3,3,1]))
    Y_list.append(np.array([1,3,3,3,2]))
    Y_list.append(np.array([1,3,3,3,3]))
    Y_list.append(np.array([2,1,1,1,1]))
    Y_list.append(np.array([2,1,1,1,2]))
    Y_list.append(np.array([2,1,1,1,3]))
    Y_list.append(np.array([2,1,1,2,1]))
    Y_list.append(np.array([2,1,1,2,2]))
    Y_list.append(np.array([2,1,1,2,3]))
    Y_list.append(np.array([2,1,1,3,1]))
    Y_list.append(np.array([2,1,1,3,2]))
    Y_list.append(np.array([2,1,1,3,3]))
    Y_list.append(np.array([2,1,2,1,1]))
    Y_list.append(np.array([2,1,2,1,2]))
    Y_list.append(np.array([2,1,2,1,3]))
    Y_list.append(np.array([2,1,2,2,1]))
    Y_list.append(np.array([2,1,2,2,2]))
    Y_list.append(np.array([2,1,2,2,3]))
    Y_list.append(np.array([2,1,2,3,1]))
    Y_list.append(np.array([2,1,2,3,2]))
    Y_list.append(np.array([2,1,2,3,3]))
    Y_list.append(np.array([2,1,3,1,1]))
    Y_list.append(np.array([2,1,3,1,2]))
    Y_list.append(np.array([2,1,3,1,3]))
    Y_list.append(np.array([2,1,3,2,1]))
    Y_list.append(np.array([2,1,3,2,2]))
    Y_list.append(np.array([2,1,3,2,3]))
    Y_list.append(np.array([2,1,3,3,1]))
    Y_list.append(np.array([2,1,3,3,2]))
    Y_list.append(np.array([2,1,3,3,3]))
    Y_list.append(np.array([2,2,1,1,1]))
    Y_list.append(np.array([2,2,1,1,2]))
    Y_list.append(np.array([2,2,1,1,3]))
    Y_list.append(np.array([2,2,1,2,1]))
    Y_list.append(np.array([2,2,1,2,2]))
    Y_list.append(np.array([2,2,1,2,3]))
    Y_list.append(np.array([2,2,1,3,1]))
    Y_list.append(np.array([2,2,1,3,2]))
    Y_list.append(np.array([2,2,1,3,3]))
    Y_list.append(np.array([2,2,2,1,1]))
    Y_list.append(np.array([2,2,2,1,2]))
    Y_list.append(np.array([2,2,2,1,3]))
    Y_list.append(np.array([2,2,2,2,1]))
    Y_list.append(np.array([2,2,2,2,2]))
    Y_list.append(np.array([2,2,2,2,3]))
    Y_list.append(np.array([2,2,2,3,1]))
    Y_list.append(np.array([2,2,2,3,2]))
    Y_list.append(np.array([2,2,2,3,3]))
    Y_list.append(np.array([2,2,3,1,1]))
    Y_list.append(np.array([2,2,3,1,2]))
    Y_list.append(np.array([2,2,3,1,3]))
    Y_list.append(np.array([2,2,3,2,1]))
    Y_list.append(np.array([2,2,3,2,2]))
    Y_list.append(np.array([2,2,3,2,3]))
    Y_list.append(np.array([2,2,3,3,1]))
    Y_list.append(np.array([2,2,3,3,2]))
    Y_list.append(np.array([2,2,3,3,3]))
    Y_list.append(np.array([2,3,1,1,1]))
    Y_list.append(np.array([2,3,1,1,2]))
    Y_list.append(np.array([2,3,1,1,3]))
    Y_list.append(np.array([2,3,1,2,1]))
    Y_list.append(np.array([2,3,1,2,2]))
    Y_list.append(np.array([2,3,1,2,3]))
    Y_list.append(np.array([2,3,1,3,1]))
    Y_list.append(np.array([2,3,1,3,2]))
    Y_list.append(np.array([2,3,1,3,3]))
    Y_list.append(np.array([2,3,2,1,1]))
    Y_list.append(np.array([2,3,2,1,2]))
    Y_list.append(np.array([2,3,2,1,3]))
    Y_list.append(np.array([2,3,2,2,1]))
    Y_list.append(np.array([2,3,2,2,2]))
    Y_list.append(np.array([2,3,2,2,3]))
    Y_list.append(np.array([2,3,2,3,1]))
    Y_list.append(np.array([2,3,2,3,2]))
    Y_list.append(np.array([2,3,2,3,3]))
    Y_list.append(np.array([2,3,3,1,1]))
    Y_list.append(np.array([2,3,3,1,2]))
    Y_list.append(np.array([2,3,3,1,3]))
    Y_list.append(np.array([2,3,3,2,1]))
    Y_list.append(np.array([2,3,3,2,2]))
    Y_list.append(np.array([2,3,3,2,3]))
    Y_list.append(np.array([2,3,3,3,1]))
    Y_list.append(np.array([2,3,3,3,2]))
    Y_list.append(np.array([2,3,3,3,3]))
    Y_list.append(np.array([3,1,1,1,1]))
    Y_list.append(np.array([3,1,1,1,2]))
    Y_list.append(np.array([3,1,1,1,3]))
    Y_list.append(np.array([3,1,1,2,1]))
    Y_list.append(np.array([3,1,1,2,2]))
    Y_list.append(np.array([3,1,1,2,3]))
    Y_list.append(np.array([3,1,1,3,1]))
    Y_list.append(np.array([3,1,1,3,2]))
    Y_list.append(np.array([3,1,1,3,3]))
    Y_list.append(np.array([3,1,2,1,1]))
    Y_list.append(np.array([3,1,2,1,2]))
    Y_list.append(np.array([3,1,2,1,3]))
    Y_list.append(np.array([3,1,2,2,1]))
    Y_list.append(np.array([3,1,2,2,2]))
    Y_list.append(np.array([3,1,2,2,3]))
    Y_list.append(np.array([3,1,2,3,1]))
    Y_list.append(np.array([3,1,2,3,2]))
    Y_list.append(np.array([3,1,2,3,3]))
    Y_list.append(np.array([3,1,3,1,1]))
    Y_list.append(np.array([3,1,3,1,2]))
    Y_list.append(np.array([3,1,3,1,3]))
    Y_list.append(np.array([3,1,3,2,1]))
    Y_list.append(np.array([3,1,3,2,2]))
    Y_list.append(np.array([3,1,3,2,3]))
    Y_list.append(np.array([3,1,3,3,1]))
    Y_list.append(np.array([3,1,3,3,2]))
    Y_list.append(np.array([3,1,3,3,3]))
    Y_list.append(np.array([3,2,1,1,1]))
    Y_list.append(np.array([3,2,1,1,2]))
    Y_list.append(np.array([3,2,1,1,3]))
    Y_list.append(np.array([3,2,1,2,1]))
    Y_list.append(np.array([3,2,1,2,2]))
    Y_list.append(np.array([3,2,1,2,3]))
    Y_list.append(np.array([3,2,1,3,1]))
    Y_list.append(np.array([3,2,1,3,2]))
    Y_list.append(np.array([3,2,1,3,3]))
    Y_list.append(np.array([3,2,2,1,1]))
    Y_list.append(np.array([3,2,2,1,2]))
    Y_list.append(np.array([3,2,2,1,3]))
    Y_list.append(np.array([3,2,2,2,1]))
    Y_list.append(np.array([3,2,2,2,2]))
    Y_list.append(np.array([3,2,2,2,3]))
    Y_list.append(np.array([3,2,2,3,1]))
    Y_list.append(np.array([3,2,2,3,2]))
    Y_list.append(np.array([3,2,2,3,3]))
    Y_list.append(np.array([3,2,3,1,1]))
    Y_list.append(np.array([3,2,3,1,2]))
    Y_list.append(np.array([3,2,3,1,3]))
    Y_list.append(np.array([3,2,3,2,1]))
    Y_list.append(np.array([3,2,3,2,2]))
    Y_list.append(np.array([3,2,3,2,3]))
    Y_list.append(np.array([3,2,3,3,1]))
    Y_list.append(np.array([3,2,3,3,2]))
    Y_list.append(np.array([3,2,3,3,3]))
    Y_list.append(np.array([3,3,1,1,1]))
    Y_list.append(np.array([3,3,1,1,2]))
    Y_list.append(np.array([3,3,1,1,3]))
    Y_list.append(np.array([3,3,1,2,1]))
    Y_list.append(np.array([3,3,1,2,2]))
    Y_list.append(np.array([3,3,1,2,3]))
    Y_list.append(np.array([3,3,1,3,1]))
    Y_list.append(np.array([3,3,1,3,2]))
    Y_list.append(np.array([3,3,1,3,3]))
    Y_list.append(np.array([3,3,2,1,1]))
    Y_list.append(np.array([3,3,2,1,2]))
    Y_list.append(np.array([3,3,2,1,3]))
    Y_list.append(np.array([3,3,2,2,1]))
    Y_list.append(np.array([3,3,2,2,2]))
    Y_list.append(np.array([3,3,2,2,3]))
    Y_list.append(np.array([3,3,2,3,1]))
    Y_list.append(np.array([3,3,2,3,2]))
    Y_list.append(np.array([3,3,2,3,3]))
    Y_list.append(np.array([3,3,3,1,1]))
    Y_list.append(np.array([3,3,3,1,2]))
    Y_list.append(np.array([3,3,3,1,3]))
    Y_list.append(np.array([3,3,3,2,1]))
    Y_list.append(np.array([3,3,3,2,2]))
    Y_list.append(np.array([3,3,3,2,3]))
    Y_list.append(np.array([3,3,3,3,1]))
    Y_list.append(np.array([3,3,3,3,2]))
    Y_list.append(np.array([3,3,3,3,3]))

    return Y_list


def checkP(P, x, y, a, T):
    prows, pcols = P.shape

    match = 0
    for t in range(T):
        for p in range(prows):
            if (x[t] == P[p][0]) and (y[t] == P[p][1]) and (a[t] == P[p][2]):
                match = 1

        if match == 0:
            return 0

        match = 0

    return 1


P = np.array(((1,2,1),(2,3,1),(3,3,1),(1,1,0),(2,2,0),(3,3,0)))

X1 = np.array((1,2,3,1,2))

Y1 = np.array((2,3,3,1,2))

A1 = np.array((1,1,1,0,0))

X = np.array((1,2,3,3,2))
Y = np.array((1,1,2,2,3))
A = np.array((1,1,1,1,1))

T = len(X)
