import numpy as np
import random
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from bot.deep_learning.models.numpy_model import Model
from utils.bot_helper import plot_training_progress
from game.game_core import Game
from bot.deep_learning.base_agent import BaseAgent

MAX_MEMORY = 100000
MAX_SAMPLE_SIZE = 1000
LEARNING_RATE = 0.01
GAMMA = 0.9
EPSILON = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.01

model_path = 'saved_model/param_numpy_model.npz'

class ParamNumpyAgent(BaseAgent):

    def __init__(self, game: Game):
        super().__init__(game)
        self.epsilon = EPSILON
        self.model = Model(28, 256, 9, LEARNING_RATE) #warning: the number of neurals in first layer must match the size of game.get_state()
        self.model.set_model_path(model_path)

    def get_state(self) -> np.ndarray:
        """
        Get the current game state and reshape it to 28x1 for model input
        example: array([1, 1, 0, 0, 0, 1, 0, ...0])
        """
        state = self.game.get_state(is_heuristic=False)
        return state.reshape(len(state), 1)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros((9, ), dtype=np.float64)
        if self.mode == "training":
            # decise to take a random action or not
            if random.random() < self.epsilon:
                # if yes pick a random action
                action[random.randint(0, 8)] = 1
            else:
                # if not model will predict the action
                action[np.argmax(self.model.forward(state)[2])] = 1
        elif self.mode == "perform":
            # always use model to predict action in pridict action / always predict
            action[np.argmax(self.model.forward(state)[2])] = 1
        return action

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
    
    def train(self):
        self.set_mode("training")

        scores = []

        while True:
            # get the current game state
            current_state = self.get_state()

            # get the move based on the state
            action = self.get_action(current_state)

            # perform action in game
            self.perform_action(action)

            # get the new state after performed action
            next_state = self.get_state()

            # get the reward of the action
            reward, game_over = self.get_reward()

            # train short memory with the action performed
            self.train_short_memory(current_state, action, reward, next_state)

            # remember the action and the reward
            self.remember(current_state, action, reward, next_state, game_over)

            # if game over then train long memory and start again
            if game_over:
                # reduce epsilon / percentage of random move
                self.epsilon *= EPSILON_DECAY
                self.epsilon = max(self.epsilon, MIN_EPSILON)

                # increase number of game and train long memory / re-train experience before start new game
                self.number_of_games += 1
                self.train_long_memory()

                if self.number_of_games % 10 == 0:
                    # save before start new game
                    self.model.save()

                # save the score to plot
                scores.append(self.get_score())
                plot_training_progress(scores)

                self.restart_game()

    def perform(self):
        self.set_mode("perform")

        while True:
            # get the current game state
            state = self.get_state()

            # get the model predict move
            action = self.get_action(state)

            # perform selected move
            self.perform_action(action)

            # check if game over or not
            _, game_over = self.get_reward()

            # restart game if game over
            if game_over:
                self.restart_game()

            # use pygame to control FPS and UPS
            self.game.clock.tick(60)
    
    def load_model(self):
        self.model.load()

if __name__ == '__main__':
    agent = ParamNumpyAgent(Game())

    mode = "training"

    if mode == "training":
        agent.train()
    elif mode == "perform":
        agent.perform()