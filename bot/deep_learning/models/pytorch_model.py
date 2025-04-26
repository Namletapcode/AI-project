import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, model_path='saved_model/param_pytorch_model.pth', load_saved_model: bool = True):
        super().__init__()
        
        self.model_path = model_path
        
        folder_path = os.path.dirname(self.model_path)
        os.makedirs(folder_path, exist_ok=True)
        
        if load_saved_model and os.path.exists(self.model_path):
            self.load()
        else:
            # generate random weight and bias with all element between -0.5 and 0.5
            self.linear1 = nn.Linear(state_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self) -> None:
        torch.save(self.state_dict(), self.model_path)
    
    def load(self) -> None:
        self.load_state_dict(torch.load(self.model_path))

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss_function = nn.MSELoss()
        
    def optimize(self, mini_batch, policy_dqn, target_dqn, gamma):
        # Unzip the mini_batch into separate variables
        states, actions, rewards, next_states, terminations = zip(*mini_batch)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)  # Sửa lỗi ở đây
        
        with torch.no_grad():
            # Compute the target Q-values using the target network using the Bellman equation
            target_q = rewards + (1 - terminations) * gamma * target_dqn(next_states).max(dim=1)[0]
            '''
                target_dqn(next_states) : torch.Tensor (torch.Size([32, 2]))
                    ==> tensor([[0, 0.1],
                                [0.05, 0.04],
                                [0.1, 0.2],
                                [0.3, 0.4]])
                            
                    .max(dim=1) : torch.return_types.max  (torch.Size([32]), torch.Size([32]))
                        ==> torch.return_types.max(values=tensor([0.1, 0.05, 0.2, 0.4]),
                                                    indices=tensor([1, 0, 1, 1]))
                                                
                    [0] : torch.Tensor (torch.Size([32]))
                        ==> tensor([0.1, 0.05, 0.2, 0.4])
            '''
            
        action_indices = torch.argmax(actions, dim=1)

        current_q = policy_dqn(states).gather(
            dim=1, 
            index=action_indices.unsqueeze(dim=1)
        ).squeeze()
        '''
            actions : torch.Tensor (torch.Size([32]))
                ==> tensor([1, 1, 1, 1, 0, 1, 1, 0], device='cuda:0')
                        
                .unsqueeze(1) : torch.Tensor (torch.Size([32, 1]))
                    ==> tensor([[1],
                                [1],
                                [1],
                                [1],
                                [0],
                                [1],
                                [1],
                                [0]], device='cuda:0')
                                        
            policy_dqn(states) : torch.Tensor (torch.Size([32, 2]))
                ==> tensor([[0, 0.1],
                            [0.05, 0.04],
                            [0.1, 0.2],
                            [0.3, 0.4]],
                            [0.6, 0.7],
                            [0.8, 0.9],
                            [0.1, 0.2],
                            [0.3, 0.4]], device='cuda:0')
                .gather(dim=1, index=actions.unsqueeze(1)) : torch.Tensor (torch.Size([32, 1]))
                    ==> tensor([[0.1],
                                [0.04],
                                [0.2],
                                [0.4],
                                [0.6],
                                [0.9],
                                [0.2],
                                [0.3]], device='cuda:0')
                .squeeze() : torch.Tensor (torch.Size([32]))
                    ==> tensor([0.1, 0.04, 0.2, 0.4, 0.6, 0.9, 0.2, 0.3], device='cuda:0')
        '''
        loss = self.loss_function(current_q, target_q)
        '''
            loss_function : torch.nn.MSELoss()
                ==> Mean Squared Error Loss
                
            current_q : torch.Tensor (torch.Size([32]))
                ==> tensor([0.1, 0.4, 0.2, 0.4], device='cuda:0')
                
            target_q : torch.Tensor (torch.Size([32]))
                ==> tensor([0.1, 0.5, 0.2, 0.4], device='cuda:0')
            return value : torch.Tensor (torch.Size([]))
            ((0.1 - 0.1) ** 2 +
             (0.4 - 0.5) ** 2 +
             (0.2 - 0.2) ** 2 +
             (0.4 - 0.4) ** 2) / 4 = 0.0025
                ==> tensor(0.0025, device='cuda:0')
        '''
        
        self.optimizer.zero_grad()  # Clear the gradients
        loss.backward()             # Backpropagation
        self.optimizer.step()       # Update the network parameters (weights and biases)

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()