#!/usr/bin/Python3
import random
from parameters import Parameters as p

class agent:
    agent_x = 0  # agent X-coordinate
    agent_y = 0  # agent Y-coordinate
    ax_init = 0  # initial agent X-coordinate
    ay_init = 0  # initial agent Y-coordinate
    state_vector = []  # Contains state information which is the input for the NN
    agent_reward = 0.0  # Agent's reward value
    goal_captured = False

    def agent_move(self, action):
        assert (action >= 0 and action < 4)

        if action == 0:  # Agent moves "left"
            self.agent_x -= 1
            while self.agent_x < 0:  # Cannot move out of bounds
                self.agent_x += 1

        if action == 1:  # Agent moves "right"
            self.agent_x += 1
            while self.agent_x > (p.x_dim-1):  # Cannot move out of bounds
                self.agent_x -= 1

        if action == 2:  # Agent moves "up"
            self.agent_y += 1
            while self.agent_y > (p.y_dim-1):  # Cannot move out of bounds
                self.agent_y -= 1

        if action == 3:  # Agent moves "down"
            self.agent_y -= 1
            while self.agent_y < 0:  # Cannot move out of bounds
                self.agent_y += 1

    def assign_acoords(self, gwx, gwy):
        self.agent_x = random.randint(0, (gwx - 1))
        self.agent_y = random.randint(0, (gwy - 1))
        self.ax_init = self.agent_x
        self.ay_init = self.agent_y
        self.state_vector = [0, 0]
        print('Agent: ', self.agent_x, ' ', self.agent_y)

    def reset_agent(self):  # Resets agent to initial position
        self.agent_x = self.ax_init
        self.agent_y = self.ay_init
        self.goal_captured = False
        self.agent_reward = 0.0

    def update_state_vec(self, x, y):  # State vector consists of x distance and y distance to target
        self.state_vector[0] = x - self.agent_x  # X-distance from agent to target
        self.state_vector[1] = y - self.agent_y  # Y-distance from agent to target

    def update_reward_NN(self, x, y):
        if self.agent_x == x and self.agent_y == y:
            self.agent_reward += 100.00  # Reward is 20 if target is captured
            self.goal_captured = True
        else:
            self.agent_reward -= 1.00  # Reward is -1 for each step taken where target isn't captured

    def update_reward_QL(self, x, y):
        if self.agent_x == x and self.agent_y == y:
            self.agent_reward = 100.00  # Reward is 20 if target is captured
            self.goal_captured = True
        else:
            self.agent_reward = -1.00  # Reward is -1 for each step taken where target isn't captured

class target:
    tx = 0  # target X-coordinate
    ty = 0  # target Y-coordinate
    tx_init = 0  # initial target X-coordinate
    ty_init = 0  # initial target Y-coordinate

    def assign_tcoords(self, gwx, gwy, agx, agy):
        self.tx = random.randint(0, (gwx - 1))
        self.ty = random.randint(0, (gwy - 1))
        while self.tx == agx or self.ty == agy:  # Target cannot be assigned to agent's position
            self.tx = random.randint(0, (gwx - 1))
            self.ty = random.randint(0, (gwy - 1))
        self.tx_init = self.tx
        self.ty_init = self.ty
        print('Target: ', self.tx, '    ', self.ty)
