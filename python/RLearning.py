import numpy as np
import time


def RLearning(gridworld, agent, par, ):
    nepisodes = 1000
    reward = gridworld.reward
    transition = gridworld.transition
    # number of states
    N = len(reward)

    # initialize q - value for all NxN state-transition pairs
    agent.q_value = agent.q_value + np.random.uniform(0, 1, size=transition.shape) / 1000

    cumr = np.zeros(nepisodes)

    for i in xrange(nepisodes):
        s = 12
        a = np.argmax(agent.q_value[s, :])
        temp = agent.q_value[s, a]

        while not gridworld.is_end(s):
            a = np.argmax(agent.q_value[s, :])
            temp = agent.q_value[s, a]
            chance = par.epsilon > np.random.uniform(0, 1)
            options = np.asarray(np.where(agent.q_value[s, :] > -float('Inf')))
            options = options.reshape(options.size)

            option_chance = int(np.ceil(np.random.uniform(0, 1) * len(options))) - 1

            a = (1 - chance) * a + chance * options[option_chance]

            #  note that in this gridworld the action is moving to the new location s'
            #  Q(s,a) is therefore a transition matrix from s to a=s'
            sn = a

            agent.q_value[s, a] = agent.q_value[s, a] + par.alpha * (
            reward[sn] + par.gamma * np.amax(agent.q_value[sn, :]) - agent.q_value[s, a])
            cumr[i] = cumr[i] + reward[sn]

            # print('current position ',s)

            s = sn

            # print('next position: ', s)
    return cumr, agent.q_value


class par:
    def __init__(self, epsilon, gamma, alpha):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha


def gridworld_plot(gridworld, position, waitkey=False):
    gridworld.print_world(position)
    if waitkey:
        if raw_input('Press anykey to continue: '):
            pass

    return


class agent(object):
    def __init__(self, transition):
        self.q_value = transition

    def print_optimalpath(self, gridworld):
        pos = gridworld.start_pos
        record = []
        record.append(str(pos[0] + 1))
        while not gridworld.is_end(pos):
            next_pos = np.argmax(self.q_value[pos])
            record.append(str(next_pos + 1))
            pos = next_pos

        print('->'.join(record))
        return


class gridworld(object):
    def __init__(self, shape):
        self.shape = shape
        self.width = shape[0]
        self.height = shape[1]
        self.len = self.width * self.height
        self.reward = np.zeros(shape).reshape((self.len))

    def set_map(self, barrier_pos, trap_pos, start_pos, target_pos):
        '''
            position value set for -1 means need for this field

        '''
        self.barrier_pos = [i - 1 for i in barrier_pos]
        self.trap_pos = [i - 1 for i in trap_pos]
        self.start_pos = [i - 1 for i in start_pos]
        self.target_pos = [i - 1 for i in target_pos]

        self.init_reward()
        self.init_worldmap(self.barrier_pos, self.trap_pos, self.target_pos)
        self.init_transition()
        self.init_end()
        return

    def init_end(self):
        self.end = self.target_pos + self.trap_pos
        return

    def is_end(self, pos):
        return pos in self.end

    def print_reward(self):
        s = []
        for i, rew in enumerate(self.reward):
            s.append(str(rew) + '\t')
            if i % self.width == self.width - 1:
                s.append('\n')
        print(''.join(list(s)))
        return

    def print_world(self, pos=0):
        '''
            Args:
                pos starts from 1
        '''
        s = list(self.worldmap_char)
        if pos > 0 and pos < self.len + 1:
            s[pos - 1] = '*'
        for i in xrange(1, self.height):
            index = (self.height - i) * self.width
            s.insert(index, '\n')

        print(''.join(s))

    def init_reward(self):
        for i, pos in enumerate(self.reward):
            self.reward[i] = -1
        for pos in self.trap_pos:
            self.reward[pos] = -100
        for pos in self.target_pos:
            self.reward[pos] = 10
        return

    def init_transition(self):
        transition = np.full([self.len, self.len], -float('inf'))
        for i in xrange(self.len):
            neighbors = self.find_neighbor(i).values()
            t = transition[i]
            for n in neighbors:
                if n not in self.barrier_pos:
                    t[n] = 0
            transition[i] = t
        self.transition = transition
        return

    def init_worldmap(self, barrier_pos, trap_pos, target_pos):
        '''
            -1,'X' stands for barrier
            0,'-'  stands for trap
            1,'O'  stands for normal position
            2,'+'  stands for target position
        '''
        worldmap_char = []
        worldmap = []
        for i in xrange(0, self.len):
            if i in barrier_pos:
                worldmap_char.append('X')
                worldmap.append(-1)
            elif i in trap_pos:
                worldmap_char.append('-')
                worldmap.append(0)
            elif i in target_pos:
                worldmap_char.append('+')
                worldmap.append(2)
            else:
                worldmap_char.append('O')
                worldmap.append(1)
        self.worldmap = worldmap
        self.worldmap_char = worldmap_char
        return

    def find_neighbor(self, pos):
        '''
            Args:
                pos: current position. start from 0
            Returns:
                dictionary of neighbors. Key is the relative position to the position given. Item is the absolute position of the world.
        '''
        neighbor = {}
        width = self.width
        height = self.height
        if (pos - width) >= 0:
            neighbor['up'] = pos - width
        if (pos + width) < width * height:
            neighbor['down'] = pos + width
        if pos % width != 0:
            neighbor['left'] = pos - 1
        if pos % width != width - 1:
            neighbor['right'] = pos + 1
        return neighbor


if __name__ == "__main__":
    gridworld = gridworld([4, 4])

    gridworld.set_map(barrier_pos=[7], trap_pos=[14, 15], start_pos=[13], target_pos=[16])

    agent = agent(gridworld.transition)
    par = par(0.3, 1, 0.1)
    gridworld.print_world(13)
    cumr, Q = RLearning(gridworld, agent, par)
    agent.print_optimalpath(gridworld)
