from parameters import Parameters as p
import numpy as np
import random

class QLearner:

    def __init__(self):
        self.nstates = p.x_dim*p.y_dim
        self.nactions = 4

        self.previous_state = 0
        self.current_state = 0

        self.qtable = np.zeros((self.nstates, self.nactions))

    def reset_qTable(self):
        self.qtable = np.zeros((self.nstates, self.nactions))

    def update_qTable(self, reward, act):
        qPrevious = self.qtable[self.previous_state, act]
        qNew = (1-p.alpha)*qPrevious + p.alpha*(reward + p.gamma*max(self.qtable[self.current_state]))
        self.qtable[self.previous_state, act] = qNew

    def epsilon_select(self):
        rvar = random.uniform(0, 1)
        act = 0
        if rvar >= p.epsilon:
            bestQ = -1000
            for i in range(4):
                if self.qtable[self.current_state, i] > bestQ:
                    bestQ = self.qtable[self.current_state, i]
                    act = i
        else:
            act = random.randint(0, 3)

        return act

    def greedy_select(self):
        bestQ = -1000
        act = 0
        for i in range(4):
            if self.qtable[self.current_state, i] > bestQ:
                bestQ = self.qtable[self.current_state, i]
                act = i

        return act

    def update_prev_state(self, ax, ay):
        self.current_state = ax + p.y_dim*ay
        self.previous_state = self.current_state

    def update_curr_state(self, ax, ay):
        self.current_state = ax + p.y_dim*ay
