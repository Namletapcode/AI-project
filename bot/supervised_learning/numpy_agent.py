import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from bot.supervised_learning.model import Model
from bot.heuristic_dodge import HeuristicDodgeBot
from game.game_core import Game
from bot.deep_learning.base_agent import BaseAgent
import numpy as np

LEARNING_RATE = 0.001

class Supervised_Agent(BaseAgent):
    def __init__(self):
        self.game = Game()
        self.model = Model(28, 256, 9, LEARNING_RATE)
        self.coach = HeuristicDodgeBot(self.game)
        self.number_of_games = 0

    def get_state(self) -> np.ndarray:
        state = self.game.get_state(is_heuristic=False)
        return state.reshape(len(state), 1)
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros((9,), dtype=np.float64)
        model_result = self.model.forward(state)[2]
        action[np.argmax(model_result)] = 1
        return action
    
    def perform_action(self, action: np.ndarray):
        self.game.take_action(action)

    def get_score(self) -> int:
        return self.game.score
    
    def get_coach_action(self) -> np.ndarray:
        state = self.game.get_state(True)
        coach_action = self.coach.get_action(state)
        return coach_action
    
    def is_game_over(self) -> bool:
        return self.game.get_reward()[1]
    
    def train_short_memory(self, state: np.ndarray, expected_action: np.ndarray):
        self.model.train(state, expected_action)

    def save(self):
        self.model.save()

    def reset_game(self):
        self.game.restart_game()
    
    def train(self, render: bool = False):

        while True:

            state = self.get_state()

            coach_action = self.get_coach_action()

            self.perform_action(coach_action, render)

            game_over = self.is_game_over()

            if not game_over:
                coach_action = coach_action.reshape(9, 1)
                self.train_short_memory(state, coach_action)

            else:
                self.number_of_games += 1
                print("Game:", self.number_of_games,"Score:", self.get_score())
                if self.number_of_games % 10 == 0:
                    self.save()
                self.reset_game()

    def perform(self, render:bool = True):
        
        while True:

            state = self.get_state()

            agent_action = self.get_action(state)

            self.perform_action(agent_action, render)

            game_over = self.is_game_over()

            if game_over:
                self.number_of_games += 1
                print("Game:", self.number_of_games,"Score:", self.get_score())
                self.reset_game()

            self.game.clock.tick(60)

if __name__ == "__main__":
    spv_agent = Supervised_Agent()
    spv_agent.perform()
