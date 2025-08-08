# REINFORCE

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

class Config:
    def __init__(self):
        self.seed = 42
        self.gamma = 0.99
        self.lr = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dims = [64, 64]
        self.batch_size = 32
        self.update_freq = 1

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU()):
        super(MLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(activation)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class REINFORCE:
    def __init__(self, env, config):
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        random.seed(config.seed)

        self.env = env
        self.config = config
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.policy_net = MLP(self.state_dim, self.config.hidden_dims, self.action_dim).to(self.config.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.config.device)
        logits = self.policy_net(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
        
    def select_greedy_action(self, state):
        state = torch.FloatTensor(state).to(self.config.device)
        logits = self.policy_net(state)
        return torch.argmax(logits).item()

    def update(self, batch):
        state_list, action_list, log_prob_list, reward_list = batch
        states = torch.FloatTensor(np.array(state_list)).to(self.config.device)
        actions = torch.LongTensor(np.array(action_list)).to(self.config.device)
        log_probs = torch.stack(log_prob_list).to(self.config.device)
        rewards = torch.FloatTensor(np.array(reward_list)).to(self.config.device)

        returns = self.compute_returns(reward_list)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = -(returns * log_probs).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_returns(self, rewards):
        returns = []
        g = 0
        for r in reversed(rewards):
            g = r + self.config.gamma * g
            returns.append(g)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32).to(self.config.device)

    def train(self, episodes=1000, max_steps=1000):
        total_rewards = []
        for episode in range(episodes):
            state, _ = self.env.reset(seed=self.config.seed+episode)
            episode_reward = 0
            state_list = []
            action_list = []
            log_prob_list = []
            reward_list = []
            for step in range(max_steps):
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state_list.append(state)
                action_list.append(action)
                log_prob_list.append(log_prob)
                reward_list.append(reward)
                if done:
                    break
                state = next_state
            total_rewards.append(episode_reward)
            print(f"Episode {episode} - Reward: {episode_reward}")
            self.update((state_list, action_list, log_prob_list, reward_list))
        return total_rewards
    
    def evaluate(self, env=None, episodes=10, max_steps=1000):
        with torch.no_grad():
            if env is None:
                env = self.env
            rewards = []
            for episode in range(episodes):
                state, _ = env.reset()
                episode_reward = 0
                for step in range(max_steps):
                    action = self.select_greedy_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    if done:
                        break
                    state = next_state
                rewards.append(episode_reward)
        return rewards

if __name__ == "__main__":
    train_env = gym.make("CartPole-v1")
    test_env = gym.make("CartPole-v1", render_mode="human")
    config = Config()
    agent = REINFORCE(train_env, config)
    rewards = agent.train(episodes=600, max_steps=1000)
    plt.plot(rewards)
    plt.show()
    agent.evaluate(test_env, episodes=10, max_steps=1000)