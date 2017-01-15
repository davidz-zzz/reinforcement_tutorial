import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio


def RLearning(rew_str,transtiton,par):

    nepisodes=1000

    # number of states
    N=len(rew_str)

    # initialize q - value for all NxN state-transition pairs
    Q=transtiton+np.random.uniform(0, 1,size=transtiton.shape)/1000

    cumr=np.zeros(nepisodes)


    for i in xrange(nepisodes):
        s=12
        a = np.argmax(Q[s, :])
        temp = Q[s, a]


        # gridworld_plotN(Q,Rew_str,s)

        while rew_str[s]!=10 and rew_str[s]!=-100:
            a = np.argmax(Q[s, :])
            temp = Q[s, a]
            chance=par.epsilon>np.random.uniform(0,1)
            options=np.where(Q[s,:] > -float('Inf'))[1]

            option_chance=int(np.ceil(np.random.uniform(0,1)*len(options)))-1

            a=(1-chance)*a + chance*options[option_chance]

            #  note that in this gridworld the action is moving to the new location s'
            #  Q(s,a) is therefore a transition matrix from s to a=s'
            sn=a

            Q[s, a] = Q[s, a] + par.alpha * (rew_str[sn] + par.gamma * np.amax(Q[sn,:])-Q[s, a])

            cumr[i]=cumr[i]+rew_str[sn]

            s=sn
            if rew_str[s]:
                print('dead end')
            # gridworld_plotN(Q, Rew_str, s)

        print(s)
    return cumr,Q

class par:
    def __init__(self,epsilon,gamma,alpha):
        self.epsilon=epsilon
        self.gamma=gamma
        self.alpha=alpha

def gridworld_plot(value,reward,position):
    N=np.math.sqrt(len(value))
    ims=np.reshape(value,6,6)




if __name__ == "__main__":
    transtiton=np.matrix([[-float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf')],
        [0, -float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf')],
        [-float('inf'), 0, -float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'),],
        [-float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf')],
        [0, -float('inf'), -float('inf'), -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf')],
        [-float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf')],
        [-float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf')],
        [-float('inf'), -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), -float('inf'), -float('inf')],
        [-float('inf'), -float('inf'), -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), -float('inf')],
        [-float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf'), -float('inf')],
        [-float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf')],
        [-float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), -float('inf'), -float('inf'), 0],
        [-float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), -float('inf'), -float('inf')],
        [-float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf'), 0, -float('inf')],
        [-float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf'), 0],
        [-float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), 0, -float('inf'), -float('inf'), 0, -float('inf')]]
                         )

    reward=np.matrix(
        [
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -100, -100, 10]
        ]
    ).reshape(16,1)

    par=par(0.3,1,0.1)

    cumr, Q = RLearning(reward, transtiton, par)

    print(cumr)
    print(Q)