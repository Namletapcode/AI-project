import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
import torch
import random
import itertools
import numpy as np
from game.game_core import Game
from bot.deep_learning.base_agent import BaseAgent
from bot.deep_learning.param_input.use_pytorch.model import Linear_QNet, QTrainer
from utils.training_visualizer import plot_training_progress

MAX_MEMORY = 100_000
BATCH_SIZE = 256
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.998
MIN_EPSILON = 0.1
NETWORK_UPDATE_FREQ = 500
TRAINING_MODE = 1
PERFORM_MODE = 2

class Agent(BaseAgent):

    def __init__(self, game: Game):
        super().__init__(game)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = Linear_QNet(28, 9).to(self.device)
        self.target_net = Linear_QNet(28, 9).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Load the weights from policy_net to target_net
        self.trainer = QTrainer(self.policy_net, lr=LEARNING_RATE, gamma=GAMMA)
        self.network_update_freq = NETWORK_UPDATE_FREQ # Update target network every 1000 steps

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        self.trainer.optimize(mini_sample, self.policy_net, self.target_net, GAMMA)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        # random moves: tradeoff exploration / exploitation
        action = torch.zeros(9, dtype=torch.long, device=self.device)
        if self.mode == TRAINING_MODE:
            # decise to take a random move or not
            if random.random() < self.epsilon:
                # if yes pick a random move
                action[random.randint(0, 8)] = 1
            else:
                # if not model will predict the move
                with torch.no_grad(): # eliminate gradient calculation
                    predicted_idx = self.policy_net(state.unsqueeze(dim=0)).squeeze().argmax()
                    action[predicted_idx] = 1
        elif self.mode == PERFORM_MODE:
            # always use model to predict move in pridict move / always predict
            with torch.no_grad(): # eliminate gradient calculation
                predicted_idx = self.policy_net(state.unsqueeze(dim=0)).squeeze().argmax()
                action[predicted_idx] = 1
        return action
    
    def get_state(self) -> np.ndarray: # get game state. example: array([1, 1, 0, 0, 0, 1, 0, ...0])
        return self.game.get_state()

def get_model_action(state: np.ndarray, model: Linear_QNet, device: torch.device) -> np.ndarray:
    """
    Get action from model for given game state
    
    Args:
        state: Game state as numpy array
        model: Trained PyTorch model
        device: torch.device to run model on
        
    Returns:
        action: One-hot encoded numpy array of length 9 representing the action
    """
    # Convert state to tensor
    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float,
        device=device
    )
    
    # Get prediction from model
    with torch.no_grad():
        # Add batch dimension and get prediction
        predicted_idx = model(state_tensor.unsqueeze(0)).squeeze().argmax()
        
    # Convert to one-hot action
    action = np.zeros(9)
    action[predicted_idx.item()] = 1
    
    return action

def train():
    game = Game()
    agent = Agent(game)
    rewards_per_episode = []
    scores_per_episode = []
    best_reward = -999999
    step_count = 0
    for episode in itertools.count():
        agent.restart_game()
        agent.number_of_games += 1
        # get old state
        current_state = agent.get_state()
        
        game_over = False
        episode_reward = 0
        episode_score = 0
        
        while not game_over and episode_reward < agent.stop_on_reward:

            # get move
            # Convert state to tensor efficiently
            current_state_tensor = torch.as_tensor(
                current_state, 
                dtype=torch.float, 
                device=agent.device
            )
            
            action = agent.get_action(current_state_tensor)
            
            agent.perform_action(action.cpu().numpy())
        
            next_state = agent.get_state()
            
            reward, game_over = agent.get_reward()
            
            episode_reward += reward
            episode_score = agent.get_score()
            
            # Convert next_state and reward to tensor efficiently
            next_state_tensor = torch.as_tensor(
                next_state, 
                dtype=torch.float, 
                device=agent.device
            )
            reward_tensor = torch.as_tensor(
                reward, 
                dtype=torch.float, 
                device=agent.device
            )
            
            if agent.mode == TRAINING_MODE:
                agent.remember(
                    current_state_tensor, action, reward_tensor, 
                    next_state_tensor, game_over)
                step_count += 1
            current_state = next_state
        rewards_per_episode.append(episode_reward)
        scores_per_episode.append(episode_score)

        if agent.mode == TRAINING_MODE:
            if episode_reward >= agent.stop_on_reward:
                print(f"Game {agent.number_of_games} finished after {episode} episodes")
                break
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.policy_net.save()
                
            # train long memory
            agent.train_long_memory()
            
            if step_count >= agent.network_update_freq:
                # update target network
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                step_count = 0

        plot_training_progress(scores_per_episode)
        agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)

def perform(model_path: str = "model/pytorch_model.pth"):
    """
    Use trained model to play game
    
    Args:
        model_path: Path to .pth model file
    """
    game = Game()
    agent = Agent(game)
    agent.mode = PERFORM_MODE
    
    # Load trained model
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    
    # Game loop
    while True:
        # Get current state
        game.clock.tick(60)
        current_state = agent.get_state()
        
        # Convert to tensor
        current_state_tensor = torch.as_tensor(
            current_state,
            dtype=torch.float,
            device=agent.device
        )
        
        # Get action from model
        action = agent.get_action(current_state_tensor)
        
        # Perform action
        agent.perform_action(action.cpu().numpy())
        
        # Check if game over
        _, game_over = agent.get_reward()
        
        if game_over:
            print(f"Game Over! Score: {agent.get_score()}")
            agent.restart_game()
            
if __name__ == '__main__':
    train()