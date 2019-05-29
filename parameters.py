
class Parameters:

    # Test Parameters
    stat_runs = 30
    x_dim = 5
    y_dim = 5
    test_count = 5
    steps = 30
    agent_method = 'QLearn'  # 'GA' for neuralnet, 'QLearn' for Q-Learner

    # GA Parameters
    generations = 200
    pop_size = 20
    prob_mut = 0.6
    prob_cross = 0.1

    # Q-Learning Parameters
    alpha = 0.1  # Learning rate
    gamma = 0.1  # Discount factor
    epsilon = 0.1  # Exploration rate
    episodes = 20  # Number of training episodes for Q learner