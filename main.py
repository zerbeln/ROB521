#!/usr/bin/Python3
import GA
import neuralnet
import agent
import QLearner
from parameters import Parameters as p
import numpy as np

def evo_net():
    g = GA.GA(); nn = neuralnet.NN(); a = agent.agent(); t = agent.target()

    # Initialize vectors and starting coordinates for agents and targets
    nn.create_NN(2, 3, 4)  # (n_inputs, n_outputs, hidden layer size)
    a.assign_acoords(p.x_dim, p.y_dim)
    t.assign_tcoords(p.x_dim, p.y_dim, a.ax_init, a.ay_init)

    # Create output files
    saveFile = open('BestFit.txt', 'w')  # Records best fitnesses
    rFile = open('SystemReward.txt', 'w')
    dataFile = open('Reliability.txt', 'w')  # Records how successful trained NN is using "best" policy

    for srun in range(p.stat_runs):
        g.create_pop(nn.n_weights)  # (policy_size)
        print('current stat run: ', srun)

        for j in range(g.population_size):  # Evaluate the initial population
            nn.get_weights(g.parent_pop[j])
            a.agent_reward = 0.00
            for tc in range(p.test_count):
                a.reset_agent(); a.goal_captured = False; k = 0

                while k < p.steps:  # Move around for certain number of steps unless target is captured
                    a.update_state_vec(t.tx, t.ty)  # Updates state input to NN
                    nn.get_inputs(a.state_vector)
                    act = nn.get_ouput()  # Get output from NN
                    a.agent_move(act)  # Agent moves
                    a.update_reward(t.tx, t.ty)

                    if a.goal_captured == True:
                        k = p.steps  # Stop iterating, target is captured
                    k += 1

            g.parent_fit[j] = 0
            g.parent_fit[j] += a.agent_reward  # Fitness is sum of agent rewards
            g.parent_fit[j] /= p.test_count

        saveFile.write('%f' % max(g.parent_fit)); saveFile.write('\t')

        # Train weights or neural network
        for i in range(p.generations):
            g.crossover(); g.mutate()  # Create new population for testing
            for j in range(g.population_size):  # Test offspring population
                nn.get_weights(g.offspring_pop[j])
                a.agent_reward = 0.00
                for tc in range(p.test_count):
                    a.reset_agent(); a.goal_captured = False; k = 0
                    while k < p.steps:  # Move around for certain number of steps unless target is captured
                        a.update_state_vec(t.tx, t.ty)  # Updates state input to NN
                        nn.get_inputs(a.state_vector)
                        act = nn.get_ouput()  # Get output from NN
                        a.agent_move(act)  # Agent moves
                        a.update_reward(t.tx, t.ty)

                        if a.goal_captured == True:
                            k = p.steps  # Stop iterating, target is captured
                        k += 1
                g.offspring_fit[j] = 0
                g.offspring_fit[j] += a.agent_reward
                g.offspring_fit[j] /= p.test_count
            g.down_select()  # Establish new parent population
            saveFile.write('%f' % g.parent_fit[0]); saveFile.write('\t')

            # Test Best Policy Found
            nn.get_weights(g.parent_pop[0]); a.reset_agent()
            a.goal_captured = False; a.agent_reward = 0.00; k = 0

            while k < p.steps:
                a.update_state_vec(t.tx, t.ty)
                nn.get_inputs(a.state_vector)
                act = nn.get_ouput()
                a.agent_move(act)
                a.update_reward(t.tx, t.ty)

                if a.goal_captured == True:
                    k = p.steps  # Stop iterating if target is captured
                k += 1
            system_reward = 0.00
            if a.goal_captured == True:
                dataFile.write('%d' % 1); dataFile.write('\t')
            if a.goal_captured == False:
                dataFile.write('%d' % 0); dataFile.write('\t')

            system_reward += a.agent_reward
            rFile.write('%f' % system_reward); rFile.write('\t')
        saveFile.write('\n'); rFile.write('\n'); dataFile.write('\n')  # New line for new stat run
    saveFile.close(); rFile.close(); dataFile.close()

def qLearn():
    a = agent.agent(); t = agent.target(); ql = QLearner.QLearner()

    # Initialize vectors and starting coordinates for agents and targets
    ql.reset_qTable()
    a.assign_acoords(p.x_dim, p.y_dim)
    t.assign_tcoords(p.x_dim, p.y_dim, a.ax_init, a.ay_init)
    ql.update_state(a.agent_x, a.agent_y)

    # Create output files
    saveFile = open('BestFit.txt', 'w')  # Records best fitnesses
    rFile = open('SystemReward.txt', 'w')
    dataFile = open('Reliability.txt', 'w')  # Records how successful trained NN is using "best" policy

    for srun in range(p.stat_runs):
        print('current stat run: ', srun)
        ql.reset_qTable()

        for ep in range(p.episodes):

            for k in range(p.steps):
                act = ql.select_move()
                a.agent_move(act)
                a.update_rewardQ(t.tx, t.ty)
                ql.update_qTable(a.agent_reward, act)
                ql.update_state(a.agent_x, a.agent_y)

            a.reset_agent()
            saveFile.write('%f' % np.max(ql.qtable[:, :])); saveFile.write('\t')  # Records max reward in Qtable


        # Test Best Policy Found
        a.reset_agent(); ql.update_state(a.agent_x, a.agent_y)
        a.goal_captured = False; a.agent_reward = 0.00; k = 0

        while k < p.steps:
            a.update_state_vec(t.tx, t.ty)
            act = ql.select_move()
            a.agent_move(act)
            a.update_reward(t.tx, t.ty)

            if a.goal_captured == True:
                k = p.steps  # Stop iterating if target is captured
            k += 1

        system_reward = 0.00
        if a.goal_captured == True:
            dataFile.write('%d' % 1); dataFile.write('\t')
        if a.goal_captured == False:
            dataFile.write('%d' % 0); dataFile.write('\t')

        system_reward += a.agent_reward
        rFile.write('%f' % system_reward); rFile.write('\t')
        saveFile.write('\n'); rFile.write('\n'); dataFile.write('\n')  # New line for new stat run
    saveFile.close(); rFile.close(); dataFile.close()


def main():
    if p.agent_method == 'EA':
        evo_net()
    elif p.agent_method == 'QLearn':
        qLearn()


main()  # Run the program
