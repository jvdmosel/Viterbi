import numpy as np
import sys
import matplotlib.pyplot as plt

class HMM:
    """HMM class represents a hidden markov model

    """

    def __init__(self, states, initial, transition, emission):
        """Constructor of HMM class, initializes new HMM object
        
        Arguments:
            states {set<str>} -- states of the HMM e.g. 'F', 'L'
            initial {dict<float>} -- initial probabilities
            transition {dict<float>} -- transition probabilties
            emission {dict<np.array>} -- emission probabilities

        Attributes:
            Q {set<str>} -- states
            I {dict<float>} -- initial probabilities
            T {dict<float>} -- transition probabilties
            E {dict<np.array>} -- emission probabilities
        """

        self.Q = states
        self.I = initial
        self.T = transition
        self.E = emission

class Viterbi:
    """Viterbi class used to calculate the viterbi algorithm
    as well as posterior probabilities
    
    """

    def __init__(self, hmm, obs):
        """Constructor of the Viterbi class
        
        Arguments:
            hmm {HMM} -- underlying hidden markov model
            obs {list<str>} -- observation sequence

        Attributes:
            hmm {HMM} -- underlying hidden markov model
            obs {list<str>} -- observation sequence
            L {int} -- number of observations
            K {int} -- number of hmm-states
            V {np.ndarray} -- viterbi variables
            ptr {np.ndarray} -- traceback matrix
            fwd {np.ndarray} -- forward matrix
            bwd {np.ndarray} -- backward matrix
            posterior {np.ndarray} -- posterior probabilities in logspace
            normalized {np.ndarray} -- posterior probabilities normalized
            ext {list<dict<tuple>>} -- extended probabilities -> all probabilities used in viterbi calculation
        """

        self.hmm = hmm
        self.obs = obs
        self.L = len(obs)
        self.K = len(hmm.Q)
        self.V = np.zeros((self.K, self.L))
        self.ptr = np.zeros((self.K, self.L), dtype=int)
        self.fwd = np.zeros((self.K, self.L))
        self.bwd = np.zeros((self.K, self.L))
        self.posterior = np.zeros((self.L, self.K))
        self.normalized = np.zeros((self.L, self.K))
        self.ext = [dict() for x in range(self.L)]

    def viterbi_algorithm(self):
        """Calculates the optimal state sequence using the viterbi algorithm
        
        Returns:
            [list<str>] -- optimal state sequence
        """
        # initialize
        # viterbi variable V[i][0] is already 0.0, no need to initialize
        for i, state in enumerate(self.hmm.Q):
            # needed for extended_traceback function
            self.ext[0].update({state: tuple((0,1))})
        for j, obs in enumerate(self.obs[1:], start = 1):
            for i, state in enumerate(self.hmm.Q):
                # addition instead of multiplication since we are in logspace
                # obs-1 since emission-probability array is 0-based, 
                # e.g. obs = 6 but index of roll 6 in emission array is 5
                arr = np.array([self.V[0][j-1] + np.log2(self.hmm.T['F'+state]) + np.log2(self.hmm.E[state][obs-1]),
                                self.V[1][j-1] + np.log2(self.hmm.T['L'+state]) + np.log2(self.hmm.E[state][obs-1])])
                # needed for extended_traceback function
                self.ext[j].update({state: tuple(abs(arr))})
                # find max_k
                self.V[i][j] = np.max(arr)
                # find arg max_k
                self.ptr[i][j-1] = np.argmax(arr)
        # index of the most probable state at observation L
        index = np.argmax(np.array(self.V[0][self.L-1], self.V[1][self.L-1]))
        # most probable state sequence list
        P = [None] * self.L
        # use hmm states attribute to map state index to string 
        P[self.L-1] = self.hmm.Q[index]
        # traceback via back pointers
        for i in range(self.L-2, -1, -1):
            index = self.ptr[index][i]
            P[i] = self.hmm.Q[index]
        return P

    def extended_traceback(self):
        """Marks uncertain positions in the sequence calculated by the viterbi algorithm using
        a form of extended traceback
        
        Returns:
            [list<str>] -- sequence with uncertain positions marked as 'X'
        """
        # extended traceback of viterbi algorithm
        # index of the most probable state at observation L
        index = np.argmax(np.array(self.V[0][self.L-1], self.V[1][self.L-1]))
        # epsilon 
        eps = 1.0
        # state sequence list
        # use X instead of F / L if decision was close
        P_ext = [None] * self.L
        # initialize traceback
        P_ext[self.L-1] = self.hmm.Q[index]
        i = self.L-2
        # while loop is needed since we are manipulating i
        while i >= 0:
            # index: current index of viterbi optimal path
            # for now state_index is a copy of index
            state_index = index
            # state is the corresponding state of state_index
            state = self.hmm.Q[index]
            # if the decision was close (absolute difference is less than epsilon)
            if np.abs(self.ext[i][state][0]-self.ext[i][state][1]) < eps:
                # condition is needed to simulate do-while
                condition = True
                # 2 cases:
                # 1. no transition was made - mark alternative path in which the transition was made
                # 2. transition was made - mark alternative path in which the transition was not made
                
                # case 1. no transition was made -> change state and state_index to alternative path
                if self.ptr[index][i] == state_index:
                    state_index = 0 if state_index == 1 else 1
                    state = 'F' if state == 'L' else 'L'
                # else case 2 -> no need to update state index
                # do-while loop
                while condition:
                    # break condition
                    # argmin since extended probabilities are absolute values of the true probabilities
                    condition =  self.ptr[index][i] != np.argmin(self.ext[i][state])
                    # update the index of the optimal path for the next iteration
                    index = self.ptr[index][i]
                    # mark position
                    P_ext[i] = 'X'
                    # update i
                    i = i-1
            # continue with usual traceback until next close decision
            index = self.ptr[index][i]
            P_ext[i] = self.hmm.Q[index]
            i = i-1
        return P_ext

    def forward_backward_algorithm(self):
        """Calculates the posterior probabilities using the forward-backward-algorithm
        and decodes the probabilities into a sequence

        Returns:
            [list<str>] -- decoded sequence using posterior probabilities
        """

        # forward step
        # initialize
        for i, state in enumerate(self.hmm.Q):
            self.fwd[i][0] = np.log(self.hmm.I[state] * self.hmm.E[state][self.obs[0]])
        for j, obs in enumerate(self.obs[1:], start = 1):
            for i, state in enumerate(self.hmm.Q):
                # recursion step
                logsum = np.NINF
                for k, prev_state in enumerate(self.hmm.Q):
                    # use of p + log(1 + exp(q-p)) trick
                    # logsum sums up temporary results
                    x = self.fwd[k][j-1] + np.log(self.hmm.T[prev_state+state])
                    logsum = x + np.log1p(np.exp(logsum-x))
                # update forward entry
                self.fwd[i][j] = np.log(self.hmm.E[state][obs-1]) + logsum

        # backward step
        for i, state in enumerate(self.hmm.Q):
            # initialization step not really needed since we are in log space
            # just for completeness
            self.bwd[i][self.L-1] = np.log(1) # 0.0
        for j, obs in reversed(list(enumerate(self.obs[:self.L-1]))):
            for i, state in enumerate(self.hmm.Q):
                # recursion
                logsum = np.NINF
                for k, next_state in enumerate(self.hmm.Q):
                    # again p + log(1 + exp(q-p)) trick with logsum
                    x = self.fwd[k][j+1] + np.log((self.hmm.T[state+next_state]) * self.hmm.E[next_state][self.obs[j+1]-1])
                    logsum = x + np.log1p(np.exp(logsum-x))
                self.bwd[i][j] = logsum
        logsum = np.NINF
        # calculate P(x) 
        for i, state in enumerate(self.hmm.Q):
            x = self.fwd[i][0] + np.log((self.hmm.I[state]) * self.hmm.E[state][self.obs[0]-1])
            logsum = x + np.log1p(np.exp(logsum-x))
        p_x = logsum
        p_decode = []
        # calculate posterior probabilities -> (f_k(i) * b_k(i)) / P(x)
        # in logspace this is equivalent to f_k(i) + b_k(i) - P(X)
        for i, state in enumerate(self.hmm.Q):
            for j in range(self.L):
                self.posterior[j][i] = self.fwd[i][j] + self.bwd[i][j] - p_x
        # decode posterior sequence using argmax and calculate normalized probabilities
        # standard exp normalization would yield NaN values
        # exp-normalize trick: exp(x_i - b) / sum(exp(x_j - b))
        # where b = max(x_j)
        for i in range(self.L):
            p_decode.append(self.hmm.Q[np.argmax(self.posterior[i])])
            b = np.max(self.posterior[i])
            y = np.exp(self.posterior[i] - b)
            self.normalized[i] = y / y.sum()
        return p_decode

def durbin(rolls, actual, viterbi):
    """Prints a comparison of the implemented viterbi algorithm with
    durbins viterbi algorithm
    
    Arguments:
        rolls {list<int>} -- sequence of die rolls
        actual {list<str>} -- actual hidden sequence just for reference
        viterbi {list<str>} -- viterbi sequence of the implemented algorithm
    """
    durbin = []
    with open('Durbin.txt', 'r') as f:
        for line in f.readlines():
            for char in line:
                durbin.append(char)
    print('\nRolls:     ' + ''.join(rolls[:60]))
    print('Die:       ' + ''.join(actual[:60]))
    print('Durbin:    ' + ''.join(durbin[:60]))
    print('Viterbi:   ' + ''.join(viterbi[:60]))
    print('\n')
    print('Rolls:     ' + ''.join(rolls[60:120]))
    print('Die:       ' + ''.join(actual[60:120]))   
    print('Durbin:    ' + ''.join(durbin[60:120]))
    print('Viterbi:   ' + ''.join(viterbi[60:120]))
    print('\n')
    print('Rolls:     ' + ''.join(rolls[120:180]))
    print('Die:       ' + ''.join(actual[120:180]))     
    print('Durbin:    ' + ''.join(durbin[120:180]))
    print('Viterbi:   ' + ''.join(viterbi[120:180]))
    print('\n')
    print('Rolls:     ' + ''.join(rolls[180:240]))
    print('Die:       ' + ''.join(actual[180:240]))    
    print('Durbin:    ' + ''.join(durbin[180:240]))
    print('Viterbi:   ' + ''.join(viterbi[180:240]))    
    print('\n')    
    print('Rolls:     ' + ''.join(rolls[240:300]))
    print('Die:       ' + ''.join(actual[240:300]))
    print('Durbin:    ' + ''.join(durbin[240:300])) 
    print('Viterbi:   ' + ''.join(viterbi[240:300]))

def posterior(rolls, actual, posterior, viterbi):
    """Prints a comparison between viterbi and the decoded posterior sequence
    
    Arguments:
        rolls {list<int>} -- sequence of die rolls
        actual {list<str>} -- actual hidden sequence just for reference
        posterior {list<str>} -- decoded posterior sequence
        viterbi {list<str>} -- viterbi sequence
    """
    print('\nRolls:     ' + ''.join(rolls[:60]))
    print('Die:       ' + ''.join(actual[:60]))
    print('Posterior: ' + ''.join(posterior[:60]))
    print('Viterbi:   ' + ''.join(viterbi[:60]))
    print('\n')
    print('Rolls:     ' + ''.join(rolls[60:120]))
    print('Die:       ' + ''.join(actual[60:120]))   
    print('Posterior: ' + ''.join(posterior[60:120]))
    print('Viterbi:   ' + ''.join(viterbi[60:120]))
    print('\n')
    print('Rolls:     ' + ''.join(rolls[120:180]))
    print('Die:       ' + ''.join(actual[120:180]))     
    print('Posterior: ' + ''.join(posterior[120:180]))
    print('Viterbi:   ' + ''.join(viterbi[120:180]))
    print('\n')
    print('Rolls:     ' + ''.join(rolls[180:240]))
    print('Die:       ' + ''.join(actual[180:240]))    
    print('Posterior: ' + ''.join(posterior[180:240]))
    print('Viterbi:   ' + ''.join(viterbi[180:240]))    
    print('\n')    
    print('Rolls:     ' + ''.join(rolls[240:300]))
    print('Die:       ' + ''.join(actual[240:300]))
    print('Posterior: ' + ''.join(posterior[240:300]))
    print('Viterbi:   ' + ''.join(viterbi[240:300]))

def marked(rolls, actual, viterbi, marked, extended):
    """Prints the marked sequences using decoded posterior and extended
    traceback sequences

    Arguments:
        rolls {list<int>} -- sequence of die rolls
        actual {list<str>} -- actual hidden sequence just for reference
        viterbi {list<str>} -- viterbi sequence
        marked {list<str>} -- marked sequence obtained by comparing viterbi and posterior sequences
        extended {list<str>} -- sequence obtained by extended traceback
    """

    print('\nRolls:     ' + ''.join(rolls[:60]))
    print('Die:       ' + ''.join(actual[:60]))
    print('Viterbi:   ' + ''.join(viterbi[:60]))
    print('Marked:    ' + ''.join(marked[:60]))
    print('Extended:  ' + ''.join(extended[:60]))
    print('\n')
    print('Rolls:     ' + ''.join(rolls[60:120]))
    print('Die:       ' + ''.join(actual[60:120]))
    print('Viterbi:   ' + ''.join(viterbi[60:120])) 
    print('Marked:    ' + ''.join(marked[60:120]))
    print('Extended:  ' + ''.join(extended[60:120]))
    print('\n')
    print('Rolls:     ' + ''.join(rolls[120:180]))
    print('Die:       ' + ''.join(actual[120:180]))
    print('Viterbi:   ' + ''.join(viterbi[120:180]))  
    print('Marked:    ' + ''.join(marked[120:180]))
    print('Extended:  ' + ''.join(extended[120:180]))
    print('\n')
    print('Rolls:     ' + ''.join(rolls[180:240]))
    print('Die:       ' + ''.join(actual[180:240]))
    print('Viterbi:   ' + ''.join(viterbi[180:240]))   
    print('Marked:    ' + ''.join(marked[180:240]))
    print('Extended:  ' + ''.join(extended[180:240]))
    print('\n')    
    print('Rolls:     ' + ''.join(rolls[240:300]))
    print('Die:       ' + ''.join(actual[240:300]))
    print('Viterbi:   ' + ''.join(viterbi[240:300]))
    print('Marked:    ' + ''.join(marked[240:300]))
    print('Extended:  ' + ''.join(extended[240:300]))

def plot_posterior(normalized):
    """Plots the posterior probabilities of the fair die
    
    Arguments:
        normalized {np.ndarray} -- normalized posterior probabilities
    """
    # new figure
    plt.figure(figsize=(16,3))
    # plot only fair posterior probability
    plt.plot(normalized[:,0])
    # limit x, y axes
    plt.xlim([0.0, 300])
    plt.ylim([0.0, 1.0])
    plt.ylabel('P(fair)',fontsize=16)    
    plt.title('Posterior probability of being in the state fair',fontsize=18)
    # save plot as png
    plt.savefig('posterior.png')
    plt.show()

def mark_positions(viterbi, posterior):
    """Mark uncertain positions by comparing decoded posterior sequence
    to the viterbi sequence
    
    Arguments:
        viterbi {list<str>} -- viterbi sequence
        posterior {list<str>} -- decoded posterior sequence
    
    Returns:
        [list<str>] -- marked sequence
    """

    seq = []
    # simply mark positions where viterbis prediction and posterior decoding differ
    # might be too naive(?)
    # possibly the length of the difference should be considered as well
    for i in range(len(viterbi)):
        if viterbi[i] != posterior[i]:
            seq.append('X')
        else:
            seq.append(viterbi[i])
    return seq

def main():
    """Reads in necessary files and loops over user input

    """

    # ocassionally dishonest casino hidden markov model
    hmm = HMM(['F', 'L'], {'F':0.5, 'L':0.5},
            {'FF':0.95, 'FL':0.05, 'LL':0.9, 'LF':0.1},
            {'F': np.repeat(1/6, 6), 'L': np.append(np.repeat(0.1, 5), 0.5)})
    obs = []
    # read in rolls
    with open('Casino.txt', 'r') as f:
        for line in f.readlines():
            for char in line:
                obs.append(int(char))
    known_seq = []
    # read in actual state sequence from the durbin book for reference
    with open('Die.txt', 'r') as f:
        for line in f.readlines():
            for char in line:
                known_seq.append(char)
    # new viterbi object
    viterbi = Viterbi(hmm, obs)
    # calculate viterbi algorithm
    viterbi_seq = viterbi.viterbi_algorithm()
    # calculate posterior probabilities
    posterior_seq = viterbi.forward_backward_algorithm()
    # extended traceback "algorithm" 
    ext_seq = viterbi.extended_traceback()
    # use posterior decoding and viterbi to mark positions
    marked_seq = mark_positions(viterbi_seq, posterior_seq)
    rolls = [str(i) for i in obs]
    # plot the posterior probability of the fair die
    plot_posterior(viterbi.normalized)
    print('\nPlease type in queries as specified by the report. \n')
    # main loop
    try:
        while True:
            query = input()
            if query == 'durbin':
                durbin(rolls, known_seq, viterbi_seq)
                print('\n' + ('-' * 71))
            elif query == 'posterior':
                posterior(rolls, known_seq, posterior_seq, viterbi_seq)
                print('\n' + ('-' * 71))
            elif query == 'marked':
                marked(rolls, known_seq, viterbi_seq, marked_seq, ext_seq)
                print('\n' + ('-' * 71))
            else:
                print('Error: ' + query + ' is not a valid input' + '\n')
    # catch interrupts, exit
    except (KeyboardInterrupt, SystemExit, EOFError):
        print('\nExiting program')
        sys.exit(0)

if __name__ == "__main__":
    main()


