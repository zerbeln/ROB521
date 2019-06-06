
class Parameters:

    # Test Parameters
    stat_runs = 30
    x_dim = 5
    y_dim = 5
    steps = 20
    agent_method = 'QLearn'  # 'EA' for neuralnet, 'QLearn' for Q-Learner

    # GA Parameters
    generations = 200
    pop_size = 40
    prob_mut = 0.6
    prob_cross = 0.1

    # NN Parameters
    ninputs = 2
    hsize = 4
    noutputs = 3

    # Q-Learning Parameters
    alpha = 0.2  # Learning rate
    gamma = 0.1  # Discount factor
    epsilon = 0.1  # Exploration rate
    episodes = 200  # Number of training episodes for Q learner