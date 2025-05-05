if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from bot.deep_learning.models.numpy_model import Model
from utils.bot_helper import plot_training_progress, get_screen_shot_gray_scale, show_numpy_to_image
from bot.deep_learning.base_agent import BaseAgent
from game.game_core import Game
import numpy as np
import random

MAX_MEMORY = 10000
MAX_SAMPLE_SIZE = 1000
LEARNING_RATE = 0.0001
GAMMA = 0.9
EPSILON = 1
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.01

IMG_SIZE = 50 # 50 x 50 pixel^2

model_path = 'saved_model/vision_numpy_model.npz'

class Agent(BaseAgent):

    def __init__(self, game: Game):
        super().__init__(game)
        self.epsillon = EPSILON
        self.model = Model((IMG_SIZE ** 2) * 2, 9, 9, LEARNING_RATE, model_path) #warning: the number of neurals in first layer must match the size of game.get_state()
        self.reset_self_img()

    def reset_self_img(self):
        self.img_01 = np.zeros((IMG_SIZE ** 2, 1), dtype=np.float64)
        self.img_02 = np.zeros((IMG_SIZE ** 2, 1), dtype=np.float64)

    def get_state(self) -> np.ndarray: # get game state. stack of two consecutive screenshot around player
        self.img_02 = get_screen_shot_gray_scale(self.game.player.x, self.game.player.y, IMG_SIZE)
        state = np.concatenate((self.img_01, self.img_02), axis=0)
        self.img_01 = self.img_02
        return state

    def get_action(self, state: np.ndarray) -> np.ndarray:
        move = np.zeros((9, ), dtype=np.float64)
        if self.mode == "train":
            # decise to take a random move or not
            if random.random() < self.epsillon:
                # if yes pick a random move
                move[random.randint(0, 8)] = 1
            else:
                # if not model will predict the move
                move[np.argmax(self.model.forward(state)[2])] = 1
        elif self.mode == "perform":
            # always use model to predict move in pridict move / always predict
            move[np.argmax(self.model.forward(state)[2])] = 1
        return move
    
    def restart_game(self):
        self.game.restart_game()
        self.reset_self_img()

    def train_short_memory(self, current_state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, game_over: bool):
        target = self.convert(current_state, action, reward, next_state, game_over)
        self.model.train(current_state, target)

    def train_long_memory(self):
        if len(self.memory) <= MAX_SAMPLE_SIZE:
            # if have not saved over 1000 states yet
            mini_sample = self.memory
        else:
            # else pick random 1000 states to re-train
            mini_sample = random.sample(self.memory, MAX_SAMPLE_SIZE)
        for current_state, action, reward, next_state, game_over in mini_sample:
            self.train_short_memory(current_state, action, reward, next_state, game_over)

    def convert(self, current_state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, game_over: bool) -> np.ndarray:
        # use simplified Bellman equation to calculate expected output
        if not game_over:
            target = self.model.forward(current_state)[2]
            Q_new = reward + GAMMA * np.max(self.model.target_forward(next_state))
            Q_new = np.clip(Q_new, -10000, 10000)
            target[np.argmax(action)] = Q_new
        else:
            target = self.model.forward(current_state)[2]
            target[np.argmax(action)] = reward
        return target
    
    def train(self, render: bool = True):
        self.set_mode("train")

        scores = []
        mean_scores = []
        sum = 0

        while True:
            # get the current game state
            current_state = self.get_state()

            # optional (on or off by comment or not the next line): show what the AI see in real-time
            show_numpy_to_image(self.img_02, IMG_SIZE)

            # get the move based on the state
            action = self.get_action(current_state)

            # perform action in game
            self.perform_action(action, render)

            # get the new state after performed action
            next_state = self.get_state()

            # get the reward of the action
            reward, game_over = self.get_reward()

            # train short memory with the action performed
            self.train_short_memory(current_state, action, reward, next_state, game_over)

            # remember the action and the reward
            self.remember(current_state, action, reward, next_state, game_over)

            # if game over then train long memory and start again
            if game_over:
                # reduce epsilon / percentage of random move
                self.epsillon *= EPSILON_DECAY
                self.epsillon = max(self.epsillon, MIN_EPSILON)

                # increase number of game and train long memory / re-train experience before start new game
                self.number_of_games += 1
                self.train_long_memory()

                if self.number_of_games % 10 == 0:
                    if self.number_of_games % 250 == 0:
                        self.model.update_target_net()

                    # save before start new game
                    agent.model.save()

                # save the score to plot
                score = self.get_score()
                sum += score
                mean_scores.append(sum / self.number_of_games)
                scores.append(score)
                plot_training_progress(scores, mean_scores)

                self.restart_game()

    def perform(self, render: bool = True):
        self.set_mode("perform")

        while True:
            # get the current game state
            state = self.get_state()

            # get the model predict move
            action = self.get_action(state)

            # perform selected move
            self.perform_action(action, render)

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
    agent = Agent(Game())
    mode = "train"

    if mode == "train":
        agent.train()
    elif mode == "perform":
        agent.perform()