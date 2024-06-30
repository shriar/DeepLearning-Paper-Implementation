# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium.spaces.box import Box
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.multiprocessing as mp
from collections import deque
import time
import math
import matplotlib.pyplot as plt

device = "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
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
    env = gym.make(env_id, max_episode_steps=max_episode_steps)
    # env = gym.make(env_id, max_episode_steps=max_episode_steps, render_mode="human")
    env = AtariRescale42x42(env)
    env = NormalizedEnv(env)
    return env


# env = create_atari_env("ALE/Pong-v5")

# %%
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
    
# model = ActorCritic(env.observation_space.shape[0], env.action_space)
# cx = torch.zeros(1, 256)
# hx = torch.zeros(1, 256)
# # model((obs, (hx, cx)))[0].shape, model((obs, (hx, cx)))[1].shape
# value, logit, (hx, cx) = model((obs, (hx, cx)))
# value.shape, logit.shape

# %%
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

gamma = 0.99
gae_lambda = 1.0
entropy_coef = 0.01
value_loss_coef = 0.5
learning_rate = 0.0001
max_grad_norm = 50
seed = 111
num_steps = 20
env = create_atari_env("ALE/Pong-v5")
shared_model = ActorCritic(env.observation_space.shape[0], env.action_space).to(device)


def train(rank, shared_model, counter, lock, optimizer=None, SaveModel=False, save_interval = 10):
    torch.manual_seed(seed + rank)
    env = create_atari_env("ALE/Pong-v5")
    # env.seed(seed + rank)
    env.unwrapped.seed(seed + rank)


    model = ActorCritic(env.observation_space.shape[0], env.action_space).to(device)
    
    if optimizer is None:
        optimizer = torch.optim.Adam(shared_model.parameters(), lr=learning_rate)

    model.train()
    
    state, _ = env.reset()
    state = torch.from_numpy(state).unsqueeze(0).to(device)
    done = True

    episode_length = 0
    while True:
        if SaveModel:
            if episode_length % save_interval == 0 and episode_length > 0:
                torch.save(shared_model.state_dict(), 'model.pth')

        episode_length += 1
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256).to(device)
            hx = torch.zeros(1, 256).to(device)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(num_steps):
            episode_length += 1
            value, logit, (hx, cx) = model((state, (hx, cx)))

            prob = F.softmax(logit, dim=-1) # [1, 6]
            log_prob = F.log_softmax(logit, dim=-1) # [1, 6]
            entropy = -(log_prob * prob).sum(1, keepdim=True) # [1, 1]
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            with lock:
                counter.value += 1

            if done:
                state, _ = env.reset()
                episode_length = 0

            state = torch.from_numpy(state).unsqueeze(0).to(device)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

                
            if done:
                break

        R = torch.zeros(1, 1).to(device)
        if not done:
            value, _, _ = model((state, (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1).to(device)

        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + gamma * values[i + 1] - values[i]
            gae = gae * gamma * gae_lambda + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - entropy_coef * entropies[i]
            loss = (policy_loss + value_loss_coef * value_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        ensure_shared_grads(model, shared_model)
        optimizer.step()
        # break


# counter = mp.Value('i', 0)
# lock = mp.Lock()
# train(0, shared_model, counter, lock, None)

# %%
# device = 'cpu'
def test(rank, shared_model, counter):
    torch.manual_seed(seed + rank)

    env = create_atari_env("ALE/Pong-v5")
    env.unwrapped.seed(seed + rank)
    model = ActorCritic(env.observation_space.shape[0], env.action_space).to(device)
    model.eval()

    state, _ = env.reset()
    state = torch.from_numpy(state).unsqueeze(0).to(device)
    reward_sum = 0
    done = True
    episode_length = 0
    actions = deque(maxlen=100)
    start_time = time.time()
    list_reawards = []

    while True:
        episode_length += 1
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256).to(device)
            hx = torch.zeros(1, 256).to(device)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state, (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1]

        state, reward, terminated, truncated, _ = env.step(action.item())
        state = torch.from_numpy(state).unsqueeze(0).to(device)
        done = terminated or truncated
        reward_sum += reward


        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True


        if done:
            print(f"Time {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start_time))}, \
                num steps {counter.value}, FPS {counter.value / (time.time() - start_time):.0f}, \
                    episode reward {reward_sum}, episode length {episode_length}")
            list_reawards.append(reward_sum)
            # plt.plot(list_reawards)
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state, _ = env.reset()
            state = torch.from_numpy(state).unsqueeze(0).to(device)
            # break
            time.sleep(30)

# test(0, shared_model, counter)    

# %%
class SharedAdam(torch.optim.Adam):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    num_processes = 15
    env = create_atari_env("ALE/Pong-v5")
    shared_model = ActorCritic(env.observation_space.shape[0], env.action_space).to(device)

    model_path = "model.pth"
    if os.path.exists(model_path):
        shared_model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully.")

    shared_model.share_memory()
    SaveModel = False
    processes = []
    learning_rate = 0.0001

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(num_processes, shared_model, counter))
    p.start()
    processes.append(p)
    optimizer = SharedAdam(shared_model.parameters(), lr=learning_rate)
    optimizer.share_memory()

    for rank in range(0, num_processes):
        if rank == 0:
            p = mp.Process(target=train, args=(rank, shared_model, counter, lock, optimizer, SaveModel, 5))
        else:
            p = mp.Process(target=train, args=(rank, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

