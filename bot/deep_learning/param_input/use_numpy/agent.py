import numpy as np
import random
from collections import deque
import sys, os, math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from bot.deep_learning.param_input.use_numpy.model import Model
from utils.training_visualizer import plot_training_progress
from game.game_core import Game
from bot.deep_learning.base_agent import BaseAgent

MAX_MEMORY = 100000
MAX_SAMPLE_SIZE = 1000
LEARNING_RATE = 0.01
GAMMA = 0.9
EPSILON = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.05
TRAINING_MODE = 1
PERFORM_MODE = 2

class Agent(BaseAgent):

    def __init__(self, game: Game):
        super().__init__(game)
        self.epsilon = EPSILON
        self.mode = TRAINING_MODE
        self.model = Model(28, 256, 9, LEARNING_RATE) #warning: the number of neurals in first layer must match the size of game.get_state()

    def get_state(self) -> np.ndarray:
        """
        Get the current game state and reshape it to 28x1 for model input
        example: array([1, 1, 0, 0, 0, 1, 0, ...0])
        """
        state = self.game.get_state()
        return state.reshape(len(state), 1)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        move = np.zeros((9, ), dtype=np.float64)
        if self.mode == TRAINING_MODE:
            # decise to take a random move or not
            if random.random() < self.epsilon:
                # if yes pick a random move
                move[random.randint(0, 8)] = 1
            else:
                # if not model will predict the move
                move[np.argmax(self.model.forward(state)[2])] = 1
        elif self.mode == PERFORM_MODE:
            # always use model to predict move in pridict move / always predict
            move[np.argmax(self.model.forward(state)[2])] = 1
        return move

    def train_short_memory(self, current_state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray):
        target = self.convert(current_state, action, reward, next_state)
        self.model.train(current_state, target)

    def train_long_memory(self):
        if len(self.memory) <= MAX_SAMPLE_SIZE:
            # if have not saved over 1000 states yet
            mini_sample = self.memory
        else:
            # else pick random 1000 states to re-train
            mini_sample = random.sample(self.memory, MAX_SAMPLE_SIZE)
        for current_state, action, reward, next_state, game_over in mini_sample:
            self.train_short_memory(current_state, action, reward, next_state)

    def convert(self, current_state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray) -> np.ndarray:
        # use simplified Bellman equation to calculate expected output
        target = self.model.forward(current_state)[2]
        Q_new = reward + GAMMA * np.max(self.model.forward(next_state)[2])
        Q_new = np.clip(Q_new, -10000, 10000)
        target[np.argmax(action)] = Q_new
        return target

def train():
    game = Game()
    agent = Agent(game)

    scores = []

    while True:
        # get the current game state
        current_state = agent.get_state()

        # get the move based on the state
        action = agent.get_action(current_state)

        # perform action in game
        agent.perform_action(action)

        # get the new state after performed action
        next_state = agent.get_state()

        # get the reward of the action
        reward, game_over = agent.get_reward()

        # train short memory with the action performed
        agent.train_short_memory(current_state, action, reward, next_state)

        # remember the action and the reward
        agent.remember(current_state, action, reward, next_state, game_over)

        # if game over then train long memory and start again
        if game_over:
            # reduce epsilon / percentage of random move
            agent.epsilon *= EPSILON_DECAY
            agent.epsilon = max(agent.epsilon, MIN_EPSILON)

            # increase number of game and train long memory / re-train experience before start new game
            agent.number_of_games += 1
            agent.train_long_memory()

            if agent.number_of_games % 10 == 0:
                # save before start new game
                # agent.model.save()
                pass

            # save the score to plot
            scores.append(agent.get_score())
            plot_training_progress(scores)

            agent.restart_game()

        agent.game.clock.tick(60)

def perform():
    game = Game()
    agent = Agent(game)
    agent.set_mode(PERFORM_MODE)

    while True:
        # get the current game state
        state = agent.get_state()

        # get the model predict move
        action = agent.get_action(state)

        # perform selected move
        agent.perform_action(action)

        # check if game over or not
        _, game_over = agent.get_reward()

        # restart game if game over
        if game_over:
            agent.restart_game()

        # use pygame to control FPS and UPS
        agent.game.clock.tick(60)


if __name__ == '__main__':
    mode = TRAINING_MODE

    if mode == TRAINING_MODE:
        train()
    elif mode == PERFORM_MODE:
        perform()