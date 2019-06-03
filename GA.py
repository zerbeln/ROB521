#!/usr/bin/Python3
import random
from parameters import Parameters as p
import numpy as np


class GA:

    def __init__(self):

        num_in_weights = (p.ninputs + 1) * p.hsize
        num_out_weights = p.noutputs * (p.hsize + 1)
        n_weights = num_in_weights + num_out_weights

        self.policy_size = n_weights
        self.population_size = p.pop_size
        self.fitness = 0.00
        self.p_mut = p.prob_mut  # Probability of mutation
        self.p_cross = p.prob_cross  # Probability of crossover
        self.policy = []  # Policy vector for GA
        self.population = np.zeros((self.population_size, self.policy_size))  # Parent population vector
        self.offspring_pop = np.zeros((self.population_size, self.policy_size))  # Offspring population
        self.pop_fit = np.zeros(self.population_size)  # vector containing fitness values for each policy in population
        self.fit_prop = np.zeros(self.population_size)  # Keeps track of probabilities in fitness proportional selection

    def create_pop(self):
        # Initialize vectors and parameters
        self.population = np.zeros((self.population_size, self.policy_size))  # Parent population vector
        self.offspring_pop = np.zeros((self.population_size, self.policy_size))  # Offspring population
        self.pop_fit = np.zeros(self.population_size)  # vector containing fitness values for each policy in population
        self.policy = np.zeros(self.policy_size)  # Creates an initial policy vector
        self.fit_prop = np.zeros(self.population_size)  # Keeps track of probabilities in fitness proportional selection

        for i in range(self.population_size):  # Randomly generate initial population
            for j in range(self.policy_size):
                self.policy[j] = random.uniform(-1, 1)
                self.population[i, j] = self.policy[j]   # Add policy to population
                self.offspring_pop[i, j] = self.policy[j]

    def mutate(self):  # Each policy in population has a certain chance of being mutated slightly
        for i in range(self.population_size):
            prob = random.uniform(0, 1)
            if prob < self.p_mut:  # Mutate with probability p_mut
                target = random.randint(0, (self.policy_size-1))
                self.population[i, target] = random.uniform(-1, 1)

    def calc_fit_proportional(self):  # Calculate the fitness probabilities for parent selection
        sum = 0.00
        for i in range(self.population_size):
            sum += (self.pop_fit[i]+500)  # Values are shifted by 500 to avoid negative numbers

        self.fit_prop = [0.00] * self.population_size  # When summed, this vector adds to 1
        assert(sum > 0)
        for j in range(self.population_size):
            if j == 0:
                self.fit_prop[j] = (self.pop_fit[j]+500) / sum
            if j > 0:
                self.fit_prop[j] = self.fit_prop[j - 1] + (self.pop_fit[j]+500)/sum


    def parent_selection(self):  # Parent's selected through fitness proportional selection
        self.calc_fit_proportional()
        p = -1
        r = random.uniform(0, 1)
        for j in range(self.population_size):
            if j == 0:
                if r < self.fit_prop[j]:
                    p = j
            if j > 0:
                if r >= self.fit_prop[j-1] and r < self.fit_prop[j]:
                    p = j
        assert(p > -1)  # A parent is always selected
        return p

    def crossover(self):
        i = 0
        while i < (self.population_size-1):
            p1 = self.parent_selection()
            p2 = self.parent_selection()
            prob = random.uniform(0,1)
            if prob < self.p_cross:  # Crossover occurs
                cp = random.randint(1,(self.policy_size-2))  # Select the crossover point
                j = 0
                while j < cp:
                    self.offspring_pop[i, j] = self.population[p1, j]
                    self.offspring_pop[i+1, j] = self.population[p2, j]
                    j += 1
                while j >= cp and j < self.policy_size:
                    self.offspring_pop[i, j] = self.population[p2, j]
                    self.offspring_pop[i+1, j] = self.population[p1, j]
                    j += 1
            else:  # Crossover does not occur
                self.offspring_pop[i, :] = self.population[p1, :]
                self.offspring_pop[i+1, :] = self.population[p2, :]
            i += 2  # Two offspring are during each iteration

        for i in range(self.population_size):
            for j in range(self.policy_size):
                self.population[i, j] = self.offspring_pop[i, j]

    def down_select(self):

        for i in range(self.population_size):
            j = i + 1
            while j < (self.population_size):
                if self.pop_fit[i] < self.pop_fit[j]:  # Re-order combined pop in terms of fitness
                    self.pop_fit[i], self.pop_fit[j] = self.pop_fit[j], self.pop_fit[i]
                    self.population[i], self.population[j] = self.population[j], self.population[i]
                j += 1
