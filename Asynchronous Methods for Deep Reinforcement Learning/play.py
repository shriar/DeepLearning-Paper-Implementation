# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium.spaces.box import Box
import cv2
import numpy as np
import time
import math


def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame


class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    def observation(self, observation):
        return _process_frame42(observation)
    
class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


max_episode_steps = 1000000
def create_atari_env(env_id):
    # env = gym.make(env_id, max_episode_steps=max_episode_steps)
    env = gym.make(env_id, max_episode_steps=max_episode_steps, render_mode="human")
    env = AtariRescale42x42(env)
    env = NormalizedEnv(env)
    return env


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        # Initialize only the output layers
        nn.init.orthogonal_(self.actor_linear.weight, gain=0.01)
        nn.init.constant_(self.actor_linear.bias, 0)
        nn.init.orthogonal_(self.critic_linear.weight, gain=1.0)
        nn.init.constant_(self.critic_linear.bias, 0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 3 * 3)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)



env = create_atari_env("ALE/Pong-v5")
state, info = env.reset()
device = "cpu"
state = torch.from_numpy(state).unsqueeze(0).float()
model = ActorCritic(env.observation_space.shape[0], env.action_space)
model.load_state_dict(torch.load('model.pth'))

cx = torch.zeros(1, 256).float()
hx = torch.zeros(1, 256).float()

for _ in range(1000):
    value, logit, (hx, cx) = model((state, (hx, cx)))
    prob = F.softmax(logit, dim=-1)
    action = prob.max(1, keepdim=True)[1]
    state, reward, terminated, truncated, info = env.step(action.item())
    state = torch.from_numpy(state).unsqueeze(0)
    env.render()
    if terminated or truncated:
        observation, info = env.reset()

env.close()