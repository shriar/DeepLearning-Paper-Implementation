import gymnasium as gym
import torch.nn  as nn
import torch
import matplotlib.pyplot as plt

env = gym.make('LunarLander-v2', render_mode="human")
# env = gym.make('LunarLander-v2', render_mode="rgb_array")
model = nn.Sequential(
    nn.Linear(env.observation_space.shape[0], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, env.action_space.n),
    # nn.Softmax(dim=1),
)
model.load_state_dict(torch.load('model_state.pth'))
obs = env.reset()[0]
done = False

def show_frame(frame):
    plt.imshow(frame)
    plt.axis('off')
    plt.show()


while not done:
    action = model(torch.Tensor(obs)).max(0)[1].item()
    # print(action)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(reward, done)
    frame = env.render()
    # show_frame(frame)
    if done:
        env.close()