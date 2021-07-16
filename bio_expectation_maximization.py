import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

pd.set_option("display.max_rows", None, "display.max_columns", None)

#Run-time instructions and output formatting

#The command line prompt accepts 3 arguments: alphabet length, number of states, and an observed input sequence
#The parameters are initialized randomly

#The program can be invoked by calling: python3 cssg28_bio_em.py alphabet_length num_states observed_sequence

#An example use case with an alphabet length of 3, 5 states, and an input sequence 0,1,2,1,0 would appear as:
#python3 cssg28_bio_em.py 3 5 0,1,2,1,0

#An alphabet length of n suggests that the input sequence can consist of integer 0 to n-1
    #E.g. with alphabet length 5, the observed input sequence can contain values 0,1,2,3,4
    #The input sequence must be delimited using commas to ensure that tokens with more than one digit (e.g. 10+) are accomodated
    #If the input sequence contains invalid values (e.g. alphabet length 5 and you supply a value of 7 in the input sequence), an error will be returned

#While the algorithm is running, progress (current iteration) is reflected on the screen every 10 iterations

#Once the algorithm exceeds the set number of iterations (by default 1000) or the termination condition is met (log(forward probability) does not improve for 5 consecutive iterations), the model parameters are returned
    #Max number of iterations may be updated on line 247
    #The initial probabilities, transition and emission matrices will print to the screen

#Implementation of Baum-Welch Expectation-Maximization (EM) algorithm to estimate HMM model parameters
    #Given a sequence of observed letters and number of states k 

#Normalize a vector (row in Numpy matrix)
def normalize(v):
    norm = np.sum(v)
    v = v/norm
    return (v, norm)

def forward(observed_sequence, tp, ep, ip):
    normalization_constants = []

    #Matrix dimensions: sequence length x number of states
    alpha = np.zeros((len(observed_sequence), len(tp[0])))
    #Populate the first row
    alpha[0, :] = ip[0:,] * ep[:, int(observed_sequence[0])]
    #Normalize first row - using Rabiner scaling (https://stats.stackexchange.com/questions/274175/scaling-step-in-baum-welch-algorithm)
    normalized = normalize(alpha[0])
    #Normalized alpha value
    alpha[0] = normalized[0]
    #Normalization constant
    norm = normalized[1]
    normalization_constants.append(norm)
    
    probs_sum = []
    #Use dynamic programming to populate the rest of alpha matrix
    for letter in range(1, len(observed_sequence)):
        for j in range(len(tp[0])):
            #Derivation of forward algorithm - http://www.adeveloperdiary.com/data-science/machine-learning/forward-and-backward-algorithm-in-hidden-markov-model/
            alpha[letter, j] = alpha[letter - 1].dot(tp[:, j]) * ep[j, int(observed_sequence[letter])]
        
        normalized = normalize(alpha[letter])
        alpha[letter] = normalized[0]
        norm = normalized[1]
        normalization_constants.append(norm)

    #Compute log(forward probability) to assess convergence
    logP = sum([math.log(c) for c in normalization_constants])
    print("LOGP", logP)

    return alpha, normalization_constants, logP

def backward(observed_sequence, tp, ep, norm_constants):

    #Intialize matrix for beta values
    beta = np.zeros((len(observed_sequence), len(tp[0])))

    #Populate the last row with 1 - working backwards by dynamic programming
    beta[len(observed_sequence) - 1] = np.ones((len(tp[0])))
    #Normalize matrix row using corresponding normalization constant from alpha timestep
    beta[len(observed_sequence) - 1] =  beta[len(observed_sequence) - 1]/norm_constants[len(observed_sequence) - 1]

    #Populate matrix
    for letter in range(len(observed_sequence) - 2, -1, -1):
        for j in range(len(tp[0])):
            #Calculation based on - http://www.adeveloperdiary.com/data-science/machine-learning/forward-and-backward-algorithm-in-hidden-markov-model/
            beta[letter, j] = (beta[letter + 1] * ep[:, int(observed_sequence[letter+1])]).dot(tp[j, :])
        #Normalize row using alpha normalization constant for corresponding timestep
        beta[letter] = beta[letter] / norm_constants[letter]

    return beta

#Format output of estimated model parameters to screen
def format_ouput(tp, ep, ip, alphabet, num_states):
    print("")
    print("_________________________________")
    print("")
    print("Alphabet: ", alphabet)
    print("Number of states: ", num_states)
    print("* Initial, transition, and emission probabilities intialized randomly")
    print("")
    print("INITIAL PROBABILITIES")
    print("* column labels denote states")
    print("")
    ip_df = pd.DataFrame(ip)
    ip_df = ip_df.rename({0: "Initial Probabilities: "}, axis='index')
    print(ip_df)
    print("")
    print("TRANSITION PROBABILITIES")
    print("* row and column labels denote states")
    print("")
    tp_df = pd.DataFrame(tp)
    print(tp_df)
    print("")
    print("EMISSION PROBABILITIES")
    print("* row labels denote states, column labels denote alphabet entries")
    print("")
    ep_df = pd.DataFrame(ep)
    print(ep_df)
    print("")
    return 1 

#Update model parameters - return updated transition and emission probabilities 
def update_params(tp, ep, gamma, xi, observed_sequence):
    
    #Update to transition and emission probability matrices; http://www.adeveloperdiary.com/data-science/machine-learning/derivation-and-implementation-of-baum-welch-algorithm-for-hidden-markov-model/
    #Updated transition probabilities = sum of all x_i / sum of all gamma up to time n-1
    tp = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
    
    #Populate gamma at the last time step
    gamma = np.hstack((gamma, np.sum(xi[:, :, len(observed_sequence) - 2], axis=0).reshape((-1, 1))))

    #Convert observed sequence to a list of ints, then to a numpy array
    observed_sequence = list(map(int, observed_sequence))
    arr = np.array(observed_sequence)

    #Update emission probabilities matrix
    #Denominator is the sum over all gamma to time n (including last row)
    denominator = np.sum(gamma, axis=1)
    #Compute numerator values 
    for l in range(len(ep[0])):
        ep[:, l] = np.sum(gamma[:, arr == l], axis=1)
    #Divide all matrix entries by denominator
    ep = np.divide(ep, denominator.reshape((-1, 1)))
    
    return tp, ep

#Parse the input observed sequence and check that it comprises only the specified alphabet
def parse_observed(observed_sequence, alphabet):
    #Split on commas
    split_on_comma = observed_sequence.split(",")
    #Convert all list entries to strings
    parsed_seq = [str(i) for i in split_on_comma]
    #Check validity of tokens - must come from the alphabet
    for entry in split_on_comma:
        if entry not in alphabet:
            print("Invalid token in observed sequence - must be contained in specified alphabet. For a supplied alphabet size n, the alphabet comprises {0,1,...n-1}. Check for duplicate commas in your input.")
            return 0
    return parsed_seq

#Randomize initial model parameters
def randomize_initial_params(tp, ep, ip, num_states, alphabet_length):
    i = 0 
    for row in tp:
        #Probabilities must sum to 1
        rand_states = np.random.dirichlet(np.ones(num_states),size=1)
        tp[i] = rand_states
        i +=1
    
    i = 0 
    for row in ep :
        rand_alphabet = np.random.dirichlet(np.ones(alphabet_length),size=1)
        ep[i] = rand_alphabet
        i +=1

    rand_states = np.random.dirichlet(np.ones(num_states),size=1)
    ip = rand_states
  
    return tp, ep, ip

#Plotting convergence
def plot(data):

    iters = []
    probs = []
    for entry in data:
        iters.append(entry[0])
        probs.append(entry[1])

    plt.plot(iters,probs)
    plt.title('Convergence Plot')
    plt.xlabel('Time Step')
    plt.ylabel('Log(Forward Probability)')
    plt.show()

#Main function 
def main():
    #Parse command line arguments
    try:
        alphabet_length = sys.argv[1] 
        num_states = sys.argv[2]
        observed_sequence = sys.argv[3]
    except:
        print("Missing input parameters - 3 arguments must be specified.")
        return 0
    #Check alphabet length is an int, and initialize the alphabet
    try:
        alphabet_length = int(alphabet_length)
        #Must be non-zero and positive
        if int(alphabet_length) <= 0:
            print("Alphabet length must be a positive integer.")
            return 0
        #Generate alphabet
        alphabet = []
        for i in range(alphabet_length):
            alphabet.append(str(i))
    except:
        print("Specified invalid alphabet length - must be an integer.")
        return 0
    #Check num states is an int
    try:
        num_states = int(num_states)
        #Must be non-zero and positive
        if int(num_states) <= 0:
            print("Number of states must be a positive integer.")
            return 0
    except:
        print("Specified invalid number of states - must be an integer.")
        return 0
    #Check observed sequence is non empty
    if len(observed_sequence) == 0:
        print("Observed sequence must be non-empty.")
        return "Error"
    #Check observed sequence only contains the alphabet
    #For now assuming the observed sequence is a string separated by commas ***CLARIFY
    parsed_seq = parse_observed(observed_sequence, alphabet)
    if parsed_seq == 0:
        return "Error"
    

    #Parameter intialization - random probabilities (ensuring that they sum to 1)
    tp = np.zeros((num_states,num_states))
    ep = np.zeros((num_states,len(alphabet)))
    ip = np.zeros((num_states))

    tp, ep, ip = randomize_initial_params(tp, ep, ip, num_states, len(alphabet))

    #Termination condition - max iterations reached or change in forward probability does not improve in 5 iterations
    max_iters = 1000
    termination_check = 0
    prev_check = False

    prev_probs = float('-inf')
    current_iter = 0
    plotting = []

    while True:

        #Log progress on the screen when many states/long seqeunce and the computation may be slow
        if current_iter > 1 and current_iter % 10 == 0:
            print("Current progress: timestep = " + str(current_iter))

        #1. Expectation step 
        #Forward algorithm
        alpha, norm_constants,probs = forward(parsed_seq, tp, ep, ip)

        probs_ = probs

        plotting.append((current_iter, probs))

        #Compute the difference in forward probability as a metric of convergence
        diff = np.abs(probs_ - prev_probs)
       
        #Check if the model is improving, else add to the termination condition
        if probs_ > prev_probs:
            prev_probs = probs_
            prev_check = False
            termination_check = 0
        else:
            if prev_check == True:
                termination_check += 1
            else:
                prev_check == True
                termination_check += 1
                
        #Backward algorithm
        beta = backward(parsed_seq, tp, ep, norm_constants)
       
        #Re-estimate x_i and gamma
        xi = np.zeros((num_states, num_states, len(parsed_seq)-1))
        
        for t in range(len(parsed_seq)-1):

            #Xi computation derived from http://www.adeveloperdiary.com/data-science/machine-learning/derivation-and-implementation-of-baum-welch-algorithm-for-hidden-markov-model/
            #Denominator = sum over all i, j (alpha_i at t * transition probability from i to j * emission probability that j emits some k * beta_j at time t+1)
            denominator = np.dot(np.dot(alpha[t, :].T, tp) * ep[:, int(parsed_seq[t + 1])].T, beta[t + 1, :])
            for i in range(num_states):
                #Numerator = alpha_i at t * transition probability from i to j * emission probability that j emits some k * beta_j at time t+1
                numerator = alpha[t, i] * tp[i, :] * ep[:, int(parsed_seq[t + 1])].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator

        #Gamma can be computed as the sum of xi values
        gamma = np.sum(xi, axis=1)
       
        #2. Maximization step
        #Calculate and update new model parameters (tp, ep) using gamma, xi
        tp, ep = update_params(tp, ep, gamma, xi, parsed_seq)
    
        #Termination condition
        if current_iter > max_iters or termination_check == 5:
            #Format output and return the initial probabilities, the transition probabilities and the emission probabilities
            format_ouput(tp, ep, ip, alphabet, num_states)
            #Plotting convergence to demonstrate correctness
            #plot(plotting)
            break
        
        current_iter += 1

    return ip, tp, ep

if __name__ == "__main__":
    main()     
