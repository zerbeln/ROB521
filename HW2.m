%Nicholas Zerbel
%ROB 521 Research Project
%06/6/2019

close all; clear all; clc

%% Neural Network Data
stat_runs = 30; %Choose which stat run to make path graph from
NN_fit_data = load('BestFit_NN.txt');
generations = 200;
x = [1:(generations+1)];
fitness = mean(NN_fit_data);
plot(x, fitness)
xlabel('Generations')
ylabel('Fitness')
title('Agent Learning Curve')

%System Reward
reward_data = load('SystemReward_NN.txt');
nn_sys_reward = mean(reward_data)

%Reliability
NN_rel = load('Reliability_NN.txt');
NN_sys_reliability = sum(NN_rel)/stat_runs;
NN_percentage_rel = NN_sys_reliability*100

%% Q-Learning Data
stat_runs = 30; %Choose which stat run to make path graph from
QL_fit_data = load('BestFit_QL.txt');
episodes = 100;
x = [1:episodes];
fitness = mean(QL_fit_data);

figure()
plot(x, fitness)
xlabel('Episodes')
ylabel('Fitness')
title('Agent Learning Curve')

%System Reward
reward_data = load('SystemReward_QL.txt');
ql_sys_reward = mean(reward_data)

%Reliability
QL_rel = load('Reliability_QL.txt');
QL_sys_reliability = sum(QL_rel)/stat_runs;
QL_percentage_rel = QL_sys_reliability*100


%% Graphs

%Reliability
XX = [1:stat_runs];
figure()
plot(XX, NN_rel)
hold on
plot(XX, QL_rel)
xlabel('Generations')
ylabel('Percentage Reliability')
title('Agent Reliability')