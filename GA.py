#!/usr/bin/Python3
import random
from parameters import Parameters as p


class GA:

    policy_size = 0
    population_size = p.pop_size
    fitness = 0.00
    p_mut = p.prob_mut  # Probability of mutation
    p_cross = p.prob_cross  # Probability of crossover
    policy = []  # Policy vector for GA
    parent_pop = []  # Parent population vector
    offspring_pop = []  # Offspring population which is created during crossover/mutation
    parent_fit = []  # vector containing fitness values for each policy in population
    offspring_fit = []  # vector containing fitness values for each policy in new population
    fit_prop = []  # Keeps track of probabilities in fitness proportional selection
    combined_pop = []  # Pools the parent and offspring pop for down-select
    combined_fit = []  # Pools fitness vectors from both populations

    def create_pop(self, pol):
        # Initialize vectors and parameters
        self.policy_size = pol
        self.parent_pop = [-1] * self.population_size
        self.offspring_pop = [-1] * self.population_size
        self.combined_pop = [-1] * (self.population_size * 2)
        self.policy = [-1] * self.policy_size  # Creates an initial policy vector
        self.parent_fit = [-1]*self.population_size  # fitness vector is same size as population
        self.offspring_fit = [-1]*self.population_size
        self.combined_fit = [-1]*(self.population_size*2)

        for i in range(self.population_size):  # Randomly generate initial population
            for j in range(self.policy_size):
                self.policy[j] = random.uniform(-1,1)
            self.parent_pop[i] = self.policy[0:self.policy_size]   # Add policy to population
            self.offspring_pop[i] = self.policy[0:self.policy_size]

    def mutate(self):  # Each policy in population has a certain chance of being mutated slightly
        for i in range(self.population_size):
            prob = random.uniform(0,1)
            if prob < self.p_mut:  # Mutate with probability p_mut
                target = random.randint(0, (self.policy_size-1))
                self.offspring_pop[i][target] = random.uniform(-1, 1)

    def calc_fit_proportional(self):  # Calculate the fitness probabilities for parent selection
        sum = 0.00
        for i in range(self.population_size):
            sum += (self.parent_fit[i]+500)  # Values are shifted by 500 to avoid negative numbers

        self.fit_prop = [0.00] * self.population_size  # When summed, this vector adds to 1
        assert(sum > 0)
        for j in range(self.population_size):
            if j == 0:
                self.fit_prop[j] = (self.parent_fit[j]+500) / sum
            if j > 0:
                self.fit_prop[j] = self.fit_prop[j - 1] + (self.parent_fit[j]+500)/sum


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
                    self.offspring_pop[i][j] = self.parent_pop[p1][j]
                    self.offspring_pop[i+1][j] = self.parent_pop[p2][j]
                    j += 1
                while j >= cp and j < self.policy_size:
                    self.offspring_pop[i][j] = self.parent_pop[p2][j]
                    self.offspring_pop[i+1][j] = self.parent_pop[p1][j]
                    j += 1
            else:  # Crossover does not occur
                self.offspring_pop[i] = self.parent_pop[p1]
                self.offspring_pop[i+1] = self.parent_pop[p2]
            i += 2  # Two offspring are during each iteration

    def down_select(self):
        self.combined_pop = self.parent_pop + self.offspring_pop  # Combine parent and offspring populations
        self.combined_fit = self.parent_fit + self.offspring_fit  # Combine parent and offspring fitnesses
        for i in  range(self.population_size*2):
            j = i + 1
            while j < (self.population_size*2):
                if self.combined_fit[i] < self.combined_fit[j]:  # Re-order combined pop in terms of fitness
                    self.combined_fit[i], self.combined_fit[j] = self.combined_fit[j], self.combined_fit[i]
                    self.combined_pop[i], self.combined_pop[j] = self.combined_pop[j], self.combined_pop[i]
                j += 1

        for k in range(self.population_size):  # Top half of combined pop becomes new parent pop
            self.parent_pop[k] = self.combined_pop[k]
            self.parent_fit[k] = self.combined_fit[k]