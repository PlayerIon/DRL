# 简单版本的NFQ

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Config:
    def __init__(self):
        self.gamma = 0.99
        self.lr = 0.001
        self.epsilon = 0.1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dims = [64, 64]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU()):
        super(MLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class NFQ:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.q_net = MLP(self.state_dim, self.config.hidden_dims, self.action_dim).to(self.config.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.lr)

    def select_action(self, state):
        if np.random.rand() < self.config.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.config.device)
            q_values = self.q_net(state)
            return torch.argmax(q_values).item()
        
    def update(self, state, action, reward, next_state, done):
        # 将数据转换为tensor并移到设备上
        state = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.config.device)
        action = torch.LongTensor([action]).to(self.config.device)
        reward = torch.FloatTensor([reward]).to(self.config.device)
        done = torch.BoolTensor([done]).to(self.config.device)
        
        # 计算当前Q值
        current_q_values = self.q_net(state)
        current_q = current_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.q_net(next_state)
            max_next_q = next_q_values.max(1)[0]
            target_q = reward + (self.config.gamma * max_next_q * (~done).float())
        
        # 计算损失
        loss = nn.MSELoss()(current_q, target_q)
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, episodes=1000, max_steps=1000):
        rewards = []
        for episode in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                if done:
                    break
            rewards.append(episode_reward)
            print(f"Episode {episode} - Reward: {episode_reward} - Epsilon: {self.config.epsilon}")
            self.config.epsilon = max(self.config.epsilon_min, self.config.epsilon * self.config.epsilon_decay)
        return rewards

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
                    action = self.select_action(state)
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
    agent = NFQ(train_env, config)
    rewards = agent.train(episodes=1000, max_steps=1000)
    plt.plot(rewards)
    plt.show()
    agent.evaluate(test_env, episodes=10, max_steps=1000)
