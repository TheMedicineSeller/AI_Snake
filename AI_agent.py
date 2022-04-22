import torch
import random
import numpy as np
from collections import deque
from Snake_game import SnakeAI, Direction, Point
from Model import Linear_QNet, QTrainer
from plotter import plot

MAX_MEMORY = 100_100
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__ (self):
        self.game_count = 0
        self.epsilon = 0  # Parameter for randomness
        self.gamma = 0.9    # Discount rate
        self.memory = deque (maxlen=MAX_MEMORY)
        self.model = Linear_QNet (11, 256, 3)
        self.trainer = QTrainer (self.model, lr=LR, gamma = self.gamma)

    def get_state (self, game):
        head = game.snake[0]
        point_l = Point (head.x - 20, head.y)
        point_r = Point (head.x + 20, head.y)
        point_u = Point (head.x, head.y - 20)
        point_d = Point (head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # The agent's state variables
        # State: 3 danger variables + OHE values(4) of snake direction + 4 food location indicators
        # Alternative :Instead of food directions, can put distance to food and try
        state = [
            # danger straight ahead
            (dir_r and game._is_collision(point_r)) or 
            (dir_l and game._is_collision(point_l)) or 
            (dir_u and game._is_collision(point_u)) or 
            (dir_d and game._is_collision(point_d)),

            # Danger right
            (dir_u and game._is_collision(point_r)) or 
            (dir_d and game._is_collision(point_l)) or 
            (dir_l and game._is_collision(point_u)) or 
            (dir_r and game._is_collision(point_d)),

            # Danger left
            (dir_d and game._is_collision(point_r)) or 
            (dir_u and game._is_collision(point_l)) or 
            (dir_r and game._is_collision(point_u)) or 
            (dir_l and game._is_collision(point_d)),
            # Movement direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < game.head.x, # food to the left bit
            game.food.x > game.head.x, # food to the right bit
            game.food.y < game.head.y, # food to the up bit
            game.food.y > game.head.y  # food to the down bit

            # Food x and y distances
            # (game.food.x - game.head.x),
            # (game.food.y - game.head.y)
        ]
        return np.array(state, dtype=int) # Converts boolean array to int (0/1)

    def get_action (self, current_state):
        # So the decisions/moves of an RL agent is essentially a tradeoff b/w exploration (randomness) and exploitation (calculated)
        # Thats why we use the randomness factor epsilon to decide the chance of agent making a random move than a trained move
        self.epsilon = 80 - self.game_count
        decision = [0, 0, 0]        # Straight, left, right
        # Hence as the game count grows in size, the probability of making a random move decreases and the if is skipped
        if random.randint(0, 200) < self.epsilon:
            move_idx = random.randint(0, 2)
            decision[move_idx] = 1
        else:
            # Convering the state array to a tensor, which is an input to the Neural net
            tensor_state = torch.tensor(current_state, dtype=torch.float)
            prediction = self.model(tensor_state)
            # Here a tensor of weights of actions is returned out of which the max weighted is made 1 (rest 0). Activation->max()
            move_idx = torch.argmax(prediction).item()
            decision[move_idx] = 1
        return decision

    def remember (self, state, action, reward, next_state, over):
        self.memory.append((state, action, reward, next_state, over)) # Appends the game knowledge as an object into the deque

    def train_long_memory (self):
        # sending to train in batches
        if len(self.memory) > BATCH_SIZE :
            sample = random.sample(self.memory, BATCH_SIZE)
        else :
            sample = self.memory
        # zip() concatenates all the variables of a type into a contiguous 'list'
        # So instead of haing a for loop over all elements of sample and calling the train function on all of them, we do this as one batch
        states, actions, rewards, next_states, overs = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, overs)

    def train_short_memory (self, state, action, reward, next_state, over):
        self.trainer.train_step(state, action, reward, next_state, over)
    
def Train ():
    plot_scores = []
    plot_meanscore = []
    total_score = 0
    high_score = 0
    
    agent = Agent()
    game = SnakeAI()
    while True:
        state = agent.get_state(game)
        next_move = agent.get_action (state)
        reward, over, score = game.play_step (next_move)
        next_state = agent.get_state (game)
        # Short memory trains the agent on the current game percepts that the snake is playing in
        agent.train_short_memory (state, next_move, reward, next_state, over)
        agent.remember (state, next_move, reward, next_state, over)

        if over :   # Plot the game result and train the agent on all previous percept history
            game.reset ()
            agent.game_count += 1
            agent.train_long_memory ()
            if high_score < score :
                high_score = score
                agent.model.save()
            print ('Game :', agent.game_count, ' Score :', score, ' High Score :', high_score)
            # plot
            plot_scores.append(score)
            total_score += score
            avg_score = total_score / agent.game_count
            plot_meanscore.append(avg_score)
            plot (plot_scores, plot_meanscore)

if __name__ == '__main__':
    Train()

