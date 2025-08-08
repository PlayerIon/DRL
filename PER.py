# 优先经验回放

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from collections import deque

class Config:
    def __init__(self):
        self.gamma = 0.99
        self.lr = 0.001
        self.epsilon = 0.1
        self.epsilon_max = 0.1
        self.epsilon_min = 0.01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dims = [64, 64]
        self.batch_size = 32
        self.replay_buffer_size = 2000
        self.update_freq = 1
        self.target_update_freq = 5
        self.tau = 0.005

        self.per_alpha = 0.6
        self.per_beta = 0.4
        self.per_beta_increment = 0.001
        self.per_epsilon = 1e-6

class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.size = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done, priority=None):
        if priority is None:
            priority = self.max_priority if self.size > 0 else 1.0
        self.max_priority = max(self.max_priority, priority)
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)

    def sample(self, batch_size):
        size = len(self.buffer)
        batch_size = min(batch_size, size)
        priorities = list(self.priorities)
        probabilities = np.array(priorities) ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(size, batch_size, p=probabilities)
        weights = (size * probabilities[indices]) ** -self.beta
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        self.beta = min(1.0, self.beta + self.beta_increment)

        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, priorities):
        for i, priority in zip(indices, priorities):
            self.priorities[i] = priority + self.epsilon
    
    def __len__(self):
        return len(self.buffer)


class DuelingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU()):
        super(DuelingMLP, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
        )
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )
        self.advantage_layer = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        feature = self.feature_layer(x)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
    
class DuelingDDQN:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.q_net = DuelingMLP(self.state_dim, self.config.hidden_dims, self.action_dim).to(self.config.device)
        self.target_q_net = DuelingMLP(self.state_dim, self.config.hidden_dims, self.action_dim).to(self.config.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.lr)
        self.replay_buffer = PrioritizedReplayBuffer(max_size=self.config.replay_buffer_size, alpha=self.config.per_alpha, \
             beta=self.config.per_beta, beta_increment=self.config.per_beta_increment, epsilon=self.config.per_epsilon)

    def select_action(self, state):
        if np.random.rand() < self.config.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.config.device)
            q_values = self.q_net(state)
            return torch.argmax(q_values).item()

    def select_greedy_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.config.device)
            q_values = self.q_net(state)
            return torch.argmax(q_values).item()

    def update(self, batch):
        states, actions, rewards, next_states, dones, weights, indices = batch
        # 将数据转换为tensor并移到设备上
        states = torch.FloatTensor(states).to(self.config.device)
        actions = torch.LongTensor(actions).to(self.config.device)
        rewards = torch.FloatTensor(rewards).to(self.config.device)
        next_states = torch.FloatTensor(next_states).to(self.config.device)
        dones = torch.BoolTensor(dones).to(self.config.device)

        current_q_values = self.q_net(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            max_next_q = self.target_q_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (self.config.gamma * max_next_q * (~dones).float())

        td_errors = torch.abs(target_q - current_q).detach()

        weights = torch.FloatTensor(weights).to(self.config.device)
        loss = (weights * nn.MSELoss()(current_q, target_q)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_errors.cpu().numpy())

        return loss.item()
        
    def update_target_network(self):
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

    def train(self, episodes=1000, max_steps=1000):
        rewards = []
        step_count = 0
        for episode in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                if step_count % self.config.update_freq == 0 and len(self.replay_buffer) > self.config.batch_size:
                    batch = self.replay_buffer.sample(self.config.batch_size)
                    self.update(batch)
                state = next_state
                episode_reward += reward
                step_count += 1
                if step_count % self.config.target_update_freq == 0:
                    self.update_target_network()
                if done:
                    break
            rewards.append(episode_reward)
            print(f"Episode {episode} - Reward: {episode_reward} - Epsilon: {round(self.config.epsilon, 3)}")
            self.adaptive_epsilon_decay(episode, episodes)
        return rewards
    
    def adaptive_epsilon_decay(self, episode, episodes):
        progress = episode / episodes
        self.config.epsilon = self.config.epsilon_min + (self.config.epsilon_max - self.config.epsilon_min) * (1 - progress)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))

    def evaluate(self, env=None, episodes=10, max_steps=1000):
        if env is None:
            env = self.env
        self.q_net.eval()
        with torch.no_grad():
            for episode in range(episodes):
                state, _ = env.reset()
                episode_reward = 0
                for step in range(max_steps):
                    action = self.select_greedy_action(state)
                    next_state, reward, done, _, _ = env.step(action)
                    state = next_state
                    episode_reward += reward
                    if done:
                        break
                print(f"Episode {episode} - Reward: {episode_reward}")

if __name__ == "__main__":
    train_env = gym.make("CartPole-v1")
    test_env = gym.make("CartPole-v1", render_mode="human")
    config = Config()
    agent = DuelingDDQN(train_env, config)
    rewards = agent.train(episodes=600, max_steps=1000)
    plt.plot(rewards)
    plt.show()
    agent.evaluate(test_env, episodes=10, max_steps=1000)
