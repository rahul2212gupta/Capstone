# mappo_gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from torch_geometric.nn import GATConv

class GNNCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super(GNNCritic, self).__init__()
        self.gat1 = GATConv(obs_dim, hidden_dim, heads=2)
        self.gat2 = GATConv(hidden_dim * 2, 1, heads=1)

    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x.squeeze()


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)


class TrajectoryBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.edge_indices = []

    def push(self, obs, action, log_prob, reward, done, edge_index):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.edge_indices.append(edge_index)

    def get(self):
        return (
            torch.tensor(np.array(self.obs), dtype=torch.float32),
            torch.tensor(self.actions, dtype=torch.long),
            torch.tensor(self.log_probs, dtype=torch.float32),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.dones, dtype=torch.float32),
            torch.tensor(np.array(self.edge_indices), dtype=torch.long).transpose(1, 0)
        )

    def clear(self):
        self.__init__()


class MAPPO_GNN:
    def __init__(self, n_agents, obs_dim, action_dim, gamma=0.99, lr=1e-4, clip_param=0.2, epochs=10):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.clip_param = clip_param
        self.epochs = epochs

        self.policy = PolicyNet(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.critic = GNNCritic(obs_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = TrajectoryBuffer()

    def select_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        probs = self.policy(obs)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), probs.detach().numpy()

    def push(self, obs, action, log_prob, reward, done, edge_index):
        self.buffer.push(obs, action, log_prob, reward, done, edge_index)

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def train_model(self):
        if len(self.buffer.obs) == 0:
            return

        obs, actions, old_log_probs, rewards, dones, edge_index = self.buffer.get()
        returns = self.compute_returns(rewards.tolist(), dones.tolist())
        values = self.critic(obs, edge_index).detach()
        advantages = returns - values

        for _ in range(self.epochs):
            probs = self.policy(obs)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            actor_loss.backward()
            self.optimizer.step()

            values = self.critic(obs, edge_index)
            critic_loss = F.mse_loss(values, returns)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        self.buffer.clear()

    def save_model(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['critic'])