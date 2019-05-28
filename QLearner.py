from parameters import Parameters as p
import numpy as np

class QLearner:
    nstates = p.x_dim*p.y_dim
    nactions = 4

    qtable = np.zeros(nstates, nactions)
