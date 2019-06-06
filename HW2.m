%Nicholas Zerbel
%ROB 521 Research Project
%06/6/2019

close all; clear all; clc

%% Neural Network Data
stat_runs = 30; %Choose which stat run to make path graph from
NN_fit_data = load('BestFit_NN.txt');
generations = 200;
x = [1:(generations)];
fitness_NN = mean(NN_fit_data);

%System Reward
reward_data = load('SystemReward_NN.txt');
nn_sys_reward = mean(reward_data)

%Reliability
NN_rel = load('Reliability_NN.txt');
NN_sys_reliability = sum(NN_rel)/stat_runs;
NN_percentage_rel = NN_sys_reliability*100

%Efficiency
NN_time_data = load('Alg_Time_NN.txt');
NN_time = sum(NN_time_data)/stat_runs

NN_steps_data = load('Steps_Taken_NN.txt');
NN_avg_steps = sum(NN_steps_data)/stat_runs

%% Q-Learning Data
stat_runs = 30; %Choose which stat run to make path graph from
QL_fit_data = load('BestFit_QL.txt');
episodes = 200;
x = [1:episodes];
fitness_QL = mean(QL_fit_data);

%System Reward
reward_data = load('SystemReward_QL.txt');
ql_sys_reward = mean(reward_data)

%Reliability
QL_rel = load('Reliability_QL.txt');
QL_sys_reliability = sum(QL_rel)/stat_runs;
QL_percentage_rel = QL_sys_reliability*100

%Efficiency
QL_time_data = load('Alg_Time_QL.txt');
QL_time = sum(QL_time_data)/stat_runs

QL_steps_data = load('Steps_Taken_QL.txt');
QL_avg_steps = sum(QL_steps_data)/stat_runs

%% Graphs

%Combined Fit
figure()
plot(x, fitness_NN)
hold on
plot(x, fitness_QL)
xlabel('Episodes')
ylabel('Fitness')
title('Agent Learning Curve')