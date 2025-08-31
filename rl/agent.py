from stable_baselines3 import PPO

class RLAgent:
    def __init__(self, env):
        self.model = PPO("MlpPolicy", env, verbose=1)

    def train(self, timesteps=20000):
        self.model.learn(total_timesteps=timesteps)

    def save(self, path="models/ppo_trader"):
        self.model.save(path)

    def load(self, path="models/ppo_trader"):
        self.model = PPO.load(path)

    def predict(self, obs):
        return self.model.predict(obs)