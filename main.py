#!/usr/bin/Python3
import GA
import neuralnet
import agent
import QLearner
from parameters import Parameters as p
import numpy as np
from time import process_time

def evo_net():
    nn = neuralnet.NN(); g = GA.GA(); a = agent.agent(); t = agent.target()

    # Initialize vectors and starting coordinates for agents and targets
    nn.create_NN(2, 3, 4)  # (n_inputs, n_outputs, hidden layer size)

    # Create output files
    learning = open('BestFit_NN.txt', 'w')  # Records best fitnesses
    perf = open('SystemReward_NN.txt', 'w')
    rel = open('Reliability_NN.txt', 'w')  # Records how successful trained NN is using "best" policy
    eff = open('Alg_Time_NN.txt', 'w')
    stp = open('Steps_Taken_NN.txt', 'w')

    for srun in range(p.stat_runs):
        print('current stat run: ', srun)
        a.assign_acoords(p.x_dim, p.y_dim)
        t.assign_tcoords(p.x_dim, p.y_dim, a.ax_init, a.ay_init)
        time_begin = process_time()
        g.create_pop()  # (policy_size)

        for j in range(g.population_size):  # Evaluate the initial population
            nn.get_weights(g.population[j])
            a.reset_agent(); k = 0

            while k < p.steps:  # Move around for certain number of steps unless target is captured
                a.update_state_vec(t.tx, t.ty)  # Updates state input to NN
                nn.get_inputs(a.state_vector)
                act = nn.get_ouput()  # Get output from NN
                a.agent_move(act)  # Agent moves
                a.update_reward_NN(t.tx, t.ty)

                if a.goal_captured == True:
                    k = p.steps  # Stop iterating, target is captured
                k += 1

            g.pop_fit[j] = a.agent_reward  # Fitness is sum of agent rewards

        learning.write('%f' % max(g.pop_fit)); learning.write('\t')

        # Train weights or neural network
        for i in range(p.generations-1):
            g.crossover(); g.mutate()  # Create new population for testing
            for j in range(g.population_size):  # Test offspring population
                nn.get_weights(g.offspring_pop[j])
                a.reset_agent(); k = 0

                while k < p.steps:  # Move around for certain number of steps unless target is captured
                    a.update_state_vec(t.tx, t.ty)  # Updates state input to NN
                    nn.get_inputs(a.state_vector)
                    act = nn.get_ouput()  # Get output from NN
                    a.agent_move(act)  # Agent moves
                    a.update_reward_NN(t.tx, t.ty)

                    if a.goal_captured == True:
                        k = p.steps  # Stop iterating, target is captured
                    k += 1

                g.pop_fit[j] = a.agent_reward

            g.down_select()  # Establish new parent population


            learning.write('%f' % g.pop_fit[0]); learning.write('\t')
        time_end = process_time()
        total_time = time_end - time_begin
        eff.write('%f' % total_time); eff.write('\n')

        # Test Best Policy Found
        nn.get_weights(g.population[0]); a.reset_agent(); k = 0
        best_fitn = max(g.pop_fit)
        assert(best_fitn == g.pop_fit[0])

        while k < p.steps:
            a.update_state_vec(t.tx, t.ty)
            nn.get_inputs(a.state_vector)
            act = nn.get_ouput()
            a.agent_move(act)
            a.update_reward_NN(t.tx, t.ty)

            if a.goal_captured == True:
                stp.write('%f' % k); stp.write('\n')
                k = p.steps  # Stop iterating if target is captured
            k += 1

        if a.goal_captured == True:
            rel.write('%d' % 1); rel.write('\t')
        else:
            rel.write('%d' % 0); rel.write('\t')

        system_reward = a.agent_reward
        perf.write('%f' % system_reward); perf.write('\t')
        learning.write('\n'); perf.write('\n'); rel.write('\n')  # New line for new stat run
    learning.close(); perf.close(); rel.close(); eff.close(); stp.close()

def qLearn():
    a = agent.agent(); t = agent.target(); ql = QLearner.QLearner()

    # Initialize vectors and starting coordinates for agents and targets
    ql.reset_qTable()

    # Create output files
    learning = open('BestFit_QL.txt', 'w')  # Records best fitnesses
    perf = open('SystemReward_QL.txt', 'w')
    rel = open('Reliability_QL.txt', 'w')  # Records how successful trained NN is using "best" policy
    eff = open('Alg_Time_QL.txt', 'w')
    stp = open('Steps_Taken_QL.txt', 'w')

    for srun in range(p.stat_runs):
        print('current stat run: ', srun)
        a.assign_acoords(p.x_dim, p.y_dim)
        t.assign_tcoords(p.x_dim, p.y_dim, a.ax_init, a.ay_init)
        time_begin = process_time()
        ql.reset_qTable()

        for ep in range(p.episodes):

            k = 0
            while k < p.steps:
                ql.update_prev_state(a.agent_x, a.agent_y)
                act = ql.epsilon_select()
                a.agent_move(act)
                ql.update_curr_state(a.agent_x, a.agent_y)
                a.update_reward_QL(t.tx, t.ty)
                ql.update_qTable(a.agent_reward, act)

                if a.goal_captured == True:
                    k = p.steps  # Stop iterating if target is captured
                k += 1

            a.reset_agent()
            learning.write('%f' % np.max(ql.qtable[:, :])); learning.write('\t')  # Records max reward in Qtable

        time_end = process_time()
        total_time = time_end - time_begin
        eff.write('%f' % total_time); eff.write('\n')

        # Test Best Policy Found
        a.reset_agent(); k = 0

        while k < p.steps:
            ql.update_prev_state(a.agent_x, a.agent_y)
            a.update_state_vec(t.tx, t.ty)
            act = ql.greedy_select()
            a.agent_move(act)
            a.update_reward_QL(t.tx, t.ty)

            if a.goal_captured == True:
                stp.write('%f' % k); stp.write('\n')
                k = p.steps  # Stop iterating if target is captured
            k += 1

        if a.goal_captured == True:  # Record reliability of agent
            rel.write('%d' % 1); rel.write('\t')
        else:
            rel.write('%d' % 0); rel.write('\t')

        system_reward = a.agent_reward  # Record system performance for stat run
        perf.write('%f' % system_reward); perf.write('\t')
        learning.write('\n'); perf.write('\n'); rel.write('\n')  # New line for new stat run
    learning.close(); perf.close(); rel.close(); stp.close()


def main():
    if p.agent_method == 'EA':
        evo_net()
    elif p.agent_method == 'QLearn':
        qLearn()


main()  # Run the program
