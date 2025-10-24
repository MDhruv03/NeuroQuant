"""
Deep Q-Network (DQN) Agent Implementation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, Optional

from config import config
from utils.logging_config import setup_logger

logger = setup_logger(__name__)


class DQNetwork(nn.Module):
    """Deep Q-Network architecture"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (256, 256, 128)):
        super(DQNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class ReplayMemory:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition"""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """Deep Q-Network Agent with experience replay and target network"""
    
    def __init__(
        self,
        env,
        learning_rate: float = None,
        gamma: float = None,
        epsilon_start: float = None,
        epsilon_end: float = None,
        epsilon_decay: float = None,
        batch_size: int = None,
        memory_size: int = None,
        target_update_freq: int = None
    ):
        """
        Initialize DQN Agent
        
        Args:
            env: Trading environment
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Minimum epsilon
            epsilon_decay: Epsilon decay rate
            batch_size: Batch size for training
            memory_size: Replay memory capacity
            target_update_freq: Frequency to update target network
        """
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters (use config defaults if not provided)
        self.learning_rate = learning_rate or config.rl.LEARNING_RATE
        self.gamma = gamma or config.rl.GAMMA
        self.epsilon = epsilon_start or config.rl.EPSILON_START
        self.epsilon_end = epsilon_end or config.rl.EPSILON_END
        self.epsilon_decay = epsilon_decay or config.rl.EPSILON_DECAY
        self.batch_size = batch_size or config.rl.BATCH_SIZE
        self.target_update_freq = target_update_freq or config.rl.TARGET_UPDATE_FREQ
        
        # Network dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Networks
        self.policy_net = DQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()
        
        # Replay memory
        memory_capacity = memory_size or config.rl.MEMORY_SIZE
        self.memory = ReplayMemory(memory_capacity)
        
        # Training stats
        self.steps = 0
        self.episodes = 0
        self.losses = []
        
        logger.info(f"DQN Agent initialized on {self.device}")
        logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode
        
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def predict(self, obs):
        """Predict action (for compatibility with other agents)"""
        action = self.select_action(obs, training=False)
        return action, None
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay memory"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, timesteps: int = None):
        """
        Train the DQN agent
        
        Args:
            timesteps: Number of training steps
        """
        timesteps = timesteps or config.training.DEFAULT_TIMESTEPS
        logger.info(f"Training DQN for {timesteps} timesteps...")
        
        episode_rewards = []
        episode_reward = 0
        
        obs, _ = self.env.reset()
        
        for step in range(timesteps):
            # Select and perform action
            action = self.select_action(obs, training=True)
            next_obs, reward, done, truncated, _ = self.env.step(action)
            
            # Store transition
            self.store_transition(obs, action, reward, next_obs, done)
            
            # Train
            loss = self.train_step()
            if loss is not None:
                self.losses.append(loss)
            
            episode_reward += reward
            obs = next_obs
            self.steps += 1
            
            # Update target network
            if self.steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            if done or truncated:
                episode_rewards.append(episode_reward)
                self.episodes += 1
                
                if self.episodes % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                    logger.info(
                        f"Episode {self.episodes}, Step {self.steps}, "
                        f"Avg Reward: {avg_reward:.2f}, "
                        f"Avg Loss: {avg_loss:.4f}, "
                        f"Epsilon: {self.epsilon:.3f}"
                    )
                
                episode_reward = 0
                obs, _ = self.env.reset()
        
        logger.info("DQN training complete")
    
    def save(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.steps = checkpoint.get('steps', 0)
        self.episodes = checkpoint.get('episodes', 0)
        logger.info(f"Model loaded from {path}")
