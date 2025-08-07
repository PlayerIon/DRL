# 使用DDQN算法，使用梯度裁剪

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

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
        self.replay_buffer_size = 10000
        self.target_update_freq = 2
        self.update_freq = 1
        self.tau = 0.005


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool_)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


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
    
class DQN:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.q_net = MLP(self.state_dim, self.config.hidden_dims, self.action_dim).to(self.config.device)
        self.target_q_net = MLP(self.state_dim, self.config.hidden_dims, self.action_dim).to(self.config.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.lr)
        self.replay_buffer = ReplayBuffer(max_size=self.config.replay_buffer_size)

    def select_action(self, state):
        if np.random.rand() < self.config.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.config.device)
            q_values = self.q_net(state)
            return torch.argmax(q_values).item()

    def select_greedy_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.config.device)
            q_values = self.q_net(state)
            return torch.argmax(q_values).item()

        
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.from_numpy(states).float().to(self.config.device)
        actions = torch.from_numpy(actions).long().to(self.config.device)
        rewards = torch.from_numpy(rewards).float().to(self.config.device)
        next_states = torch.from_numpy(next_states).float().to(self.config.device)
        dones = torch.from_numpy(dones).bool().to(self.config.device)

        current_q_values = self.q_net(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            max_next_q = self.target_q_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (self.config.gamma * max_next_q * (~dones).float())

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()
        
    def update_target_network(self):
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

    def train(self, episodes=1000, max_steps=1000):
        rewards = []
        losses = []
        step_count = 0
        for episode in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            update_count = 0
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                if step_count % self.config.update_freq == 0 and len(self.replay_buffer) > self.config.batch_size:
                    batch = self.replay_buffer.sample(self.config.batch_size)
                    loss = self.update(batch)
                    episode_loss += loss
                    update_count += 1
                state = next_state
                episode_reward += reward
                step_count += 1
                if step_count % self.config.target_update_freq == 0:
                    self.update_target_network()
                if done:
                    break
            rewards.append(episode_reward)
            avg_loss = episode_loss / max(update_count, 1)
            losses.append(avg_loss)
            print(f"Episode {episode} - Reward: {episode_reward} - Epsilon: {round(self.config.epsilon, 3)} - Avg Loss: {avg_loss:.4f}")
            self.adaptive_epsilon_decay(episode, episodes)
        return rewards, losses
    
    def adaptive_epsilon_decay(self, episode, episodes):
        progress = episode / episodes
        self.config.epsilon = self.config.epsilon_max - (self.config.epsilon_max - self.config.epsilon_min) * progress

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
    agent = DQN(train_env, config)
    rewards, losses = agent.train(episodes=2000, max_steps=1000)
    
    # 绘制奖励和损失曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(rewards)
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    
    ax2.plot(losses)
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.show()
    
    agent.evaluate(test_env, episodes=10, max_steps=1000)
