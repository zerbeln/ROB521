%Nicholas Zerbel
%ROB 538 HW 2
%10/9/2018

close all; clear all; clc

%%Fitness Data and Single Agent Learning
stat_runs = 30; %Choose which stat run to make path graph from
fit_data = load('BestFit.txt');
generations = 200;
x = [1:(generations+1)];
XX = [1:generations];
fitness = mean(fit_data);
plot(x, fitness)
xlabel('Generations')
ylabel('Fitness')
title('Neural Network Learning Curve')

%%System Reward
reward_data = load('SystemReward.txt');
sys_reward = mean(reward_data);
%figure()
%plot(XX,sys_reward)
