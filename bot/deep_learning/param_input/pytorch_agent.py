import torch
import random
import itertools
import numpy as np

if __name__ == "__main__":
    # only re-direct below if running this file
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from game.game_core import Game
from bot.deep_learning.base_agent import BaseAgent
from bot.deep_learning.models.pytorch_model import Linear_QNet, QTrainer
from bot.heuristic_dodge import HeuristicDodgeBot
from configs.bot_config import DodgeAlgorithm
from utils.bot_helper import plot_training_progress

MAX_MEMORY = 100_000
BATCH_SIZE = 256
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.998
MIN_EPSILON = 0.1
NETWORK_UPDATE_FREQ = 500

HEURISTIC_METHOD = DodgeAlgorithm.LEAST_DANGER_PATH
IMITATION_PROBABILITY = 0.2 # 20% action selected based on heuristic_bot

model_path = 'saved_model/param_pytorch_model.pth'

class ParamTorchAgent(BaseAgent):

    def __init__(self, game: Game, load_saved_model: bool = False):
        super().__init__(game)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = Linear_QNet(
            28, 9, model_path=model_path, load_saved_model=load_saved_model
            ).to(self.device)
        #warning: the number of neurals in first layer must match the size of game.get_state()
        
        self.target_net = Linear_QNet(28, 9, load_saved_model=False).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Load the weights from policy_net to target_net
        
        self.trainer = QTrainer(self.policy_net, lr=LEARNING_RATE, gamma=GAMMA)
        self.network_update_freq = NETWORK_UPDATE_FREQ # Update target network every 1000 steps
        
        self.heuristic_bot = HeuristicDodgeBot(game, HEURISTIC_METHOD)
        self.imitation_prob = IMITATION_PROBABILITY
        
    def train(self, render: bool = False) -> None:
        self.set_mode("train")
        rewards_per_episode = []
        scores_per_episode = []
        best_reward = -999999
        step_count = 0
        for episode in itertools.count():
            self.restart_game()
            self.number_of_games += 1
            # get old state
            current_state = self.get_state()
            
            game_over = False
            episode_reward = 0
            episode_score = 0
            
            while not game_over and episode_reward < self.stop_on_reward:

                # get move
                # Convert state to tensor efficiently
                current_state_tensor = torch.as_tensor(
                    current_state, 
                    dtype=torch.float, 
                    device=self.device
                )
                
                action = self.get_action(current_state)
                
                action_tensor = torch.as_tensor(
                    action,
                    dtype=torch.float,
                    device=self.device
                )
                
                self.perform_action(action, render)
            
                next_state = self.get_state()
                
                reward, game_over = self.get_reward()
                
                episode_reward += reward
                episode_score = self.get_score()
                
                # Convert next_state and reward to tensor efficiently
                next_state_tensor = torch.as_tensor(
                    next_state, 
                    dtype=torch.float, 
                    device=self.device
                )
                reward_tensor = torch.as_tensor(
                    reward, 
                    dtype=torch.float, 
                    device=self.device
                )

                self.remember(
                    current_state_tensor, action_tensor, reward_tensor, 
                    next_state_tensor, game_over)
                step_count += 1
                current_state = next_state
            rewards_per_episode.append(episode_reward)
            scores_per_episode.append(episode_score)

            if episode_reward >= self.stop_on_reward:
                print(f"Game {self.number_of_games} finished after {episode} episodes")
                break
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.policy_net.save()
                
            # train long memory
            self.train_long_memory()
            
            if step_count >= self.network_update_freq:
                # update target network
                self.target_net.load_state_dict(self.policy_net.state_dict())
                step_count = 0

            plot_training_progress(scores_per_episode)
            self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)
    
    def perform(self, render: bool = True):
        """
        Use trained model to play game
        
        Args:
            model_path: Path to .pth model file
        """
        self.set_mode("perform")
        
        # Load trained model
        self.load_model()
        # Switch the model to evaluation mode to disable dropout/batchnorm etc.
        self.policy_net.eval()
        
        # Game loop
        while True:
            # Get current state
            self.game.clock.tick(60)
            current_state = self.get_state()
            
            # Get action from model
            action = self.get_action(current_state)
            
            # Perform action
            self.perform_action(action, render)
            
            # Check if game over
            _, game_over = self.get_reward()
            
            if game_over:
                print(f"Game Over! Score: {self.get_score()}")
                self.restart_game()
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.as_tensor(
            state, 
            dtype=torch.float, 
            device=self.device
        )
        action = np.zeros(9, dtype=np.float32)
        # random moves: tradeoff exploration / exploitation
        if self.mode == "train":
            # decise to take a random move or not
            if random.random() < self.epsilon:
                if random.random() < self.imitation_prob:
                    action = self.heuristic_bot.get_action(self.game.get_state(is_heuristic=True))
                else:
                    # if yes pick a random move
                    action[random.randint(0, 8)] = 1
            else:
                # if not model will predict the move
                with torch.no_grad(): # eliminate gradient calculation
                    predicted_idx = self.policy_net(state_tensor.unsqueeze(dim=0)).squeeze().argmax()
                    action[predicted_idx] = 1
        elif self.mode == "perform":
            # always use model to predict move in pridict move / always predict
            with torch.no_grad(): # eliminate gradient calculation
                predicted_idx = self.policy_net(state_tensor.unsqueeze(dim=0)).squeeze().argmax()
                action[predicted_idx] = 1
        return action

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        self.trainer.optimize(mini_sample, self.policy_net, self.target_net, GAMMA)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_state(self) -> np.ndarray:
        return self.game.get_state(is_heuristic=False)
    
    def load_model(self):
        self.policy_net.load()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if self.mode == "perform":
            self.policy_net.eval()

if __name__ == '__main__':
    game = Game()
    agent = ParamTorchAgent(game)
    agent.train()