import time
import datetime
import random
from collections import namedtuple
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


"""

Section 1: Creating a Model Manager to Train an LSTM

"""


class ModelManager:
    """
    This class is a manager for training an LSTM with a generated integer dataset.
    Train and Test the LSTM by interacting with the methods of this class.
    By iteratively calling the traing and test methods, the algorithm will become more accurate.
    """


    """
    Constants:

    num_epochs -- The number of iterations to review the full contents of the dataset.
    num_progress -- The number of iterations to complete before printing a progress update.
    time_steps -- The length of the integer sequence in the dataset.
    batch_size -- The number of training batches to complete simultaneously. Keep at 1 for now.
    input_size -- The number of input units to the LSTM.
    hidden_size -- The number of hidden units in the LSTM per layer.
    output_size -- The number of output units in the LSTM.
    num_hidden_layers -- The number of hidden layers in the LSTM.
    gradient_threshold -- The acceptable amount of loss from which the program will stop executing.
    gamma -- The discount factor given toError future rewards.
    """
    num_epochs = 2000
    num_progress = 50
    time_steps = 2
    batch_size = 1
    action_size = 11
    state_size = 200
    input_size = action_size + state_size
    hidden_size = input_size * 2
    output_size = 1
    num_hidden_layers = 2
    gradient_threshold = 0.05
    gamma = 0.999


    class Model(nn.Module):
        """
        This model contains an LSTM created using PyTorch.
        This model will accept a batch of inputs, and will output a batch of predictions.
        """


        def reset(self):
            """
            Reset the hidden state and the cell of the LSTM.
            This is required when finishing a training batch on a single sequence.
            """
            self.h = [Variable(Tensor(ModelManager.batch_size, ModelManager.hidden_size).zero_()) for i in range(ModelManager.num_hidden_layers)]
            self.c = self.h


        def __init__(self):
            """
            Initialize the LSTM layers using PyTorch.
            Each LSTM layer is of size HIDDEN_SIZE.
            The output Layer is of size OUTPUT_SIZE.
            """
            super(ModelManager.Model, self).__init__()
            self.hidden_layers = [nn.LSTMCell((ModelManager.input_size if not i else ModelManager.hidden_size), ModelManager.hidden_size) for i in range(ModelManager.num_hidden_layers)]
            self.output_layer = nn.Linear(ModelManager.hidden_size, ModelManager.output_size)
            self.reset()


        def forward(self, x):
            """
            This function requires a batch of inputs, such as the binary encoding of an ascii character,
            and produces a batch of predictions. Both the input and output of this method are PyTorch Variables.

            x -- A PyTorch Variable of 2 dimensions representing a batch of input units.
            """
            for i in range(ModelManager.num_hidden_layers):
                self.h[i], self.c[i] = self.hidden_layers[i]((x if not i else self.h[i - 1]), (self.h[i], self.c[i]))
            model_output = self.output_layer(self.h[-1])
            return model_output


    def int_to_tensor(n):
        """
        This function will convert from an integer n into a binary tensor of 1 dimension,
        where the number of ones in the tensor indicate n.

        n -- An integer to be encoded as a binary tensor.
        """
        if n > 100: n = 100
        elif n < 0: n = 0
        return torch.cat([torch.ones(n), torch.zeros(100 - n)])


    def tensor_to_int(t):
        """
        This function will convert from a binary tensor of 1 dimension into an integer,
        where the number of ones (above 0.5) in the tensor indicate the integer to return.

        t -- A tensor of 1 dimension containing n ones (above 0.5), representing integer n.
        """
        return sum([1 if (t.data[i] >= 0.5) else 0 for i in range(100)])


    def create_model(self):
        """
        This function creates the required PyTorch objects to train an LSTM.
        """
        self.model = ModelManager.Model()
        self.optimizer = optim.Adam(self.model.parameters())


    def create_dataset(self):
        """
        This function creates the dataset of inputs and labels,
        from which to train the LSTM using supervised learning.
        """
        self.inputs = Variable(torch.stack([
            torch.cat(
                [ModelManager.int_to_tensor(i), ModelManager.int_to_tensor((ModelManager.time_steps - i))]
            ).unsqueeze(0) for i in range(ModelManager.time_steps)
        ]))
        self.labels = Variable(torch.rand(ModelManager.time_steps))


    def clear(self):
        """
        This function will reset the ModelManager class
        by recreating all objects used by this class.
        Effectively calles __init__ again.
        """
        self.create_model()
        self.start_time = time.time()
        self.data_points = []


    def __init__(self):
        """
        Initialize the model manager object.
        Create the model and dataset.
        """
        torch.manual_seed(1)
        self.create_model()
        self.create_dataset()
        self.start_time = time.time()
        self.data_points = []


    def train(self, check_progress=False):
        """
        This function will perform multiple epochs of training on the MODEL to calculate a reward value given a state and an action.
        First, the algorithm calculates an expected reward from taking the ACTION given
        the CURRENT_STATE. This calculation is compared with the actual REWARD, and calculates
        an expected future reward from the NEXT_STATE.

        check_progress -- When this Variable(is set to True, calling train will output
            the Huber Loss, Elapsed Time, and Time Remaining every NUM_PROGRESS cycles.
        """
        for i in range(ModelManager.num_epochs):
            action_list = torch.stack([
                self.argmax_action(self.inputs[j][0]) for j in range(ModelManager.time_steps)
            ])
            expected_reward = torch.stack([
                self.model(torch.cat([self.inputs[j][0], action_list[j]]).unsqueeze(0)) for j in range(ModelManager.time_steps)
            ])
            actual_reward = torch.stack([
                (ModelManager.gamma * self.max_action(self.inputs[min(j + 1, ModelManager.time_steps - 1)][0]) + self.labels[j]) for j in range(ModelManager.time_steps)
            ])
            loss = F.smooth_l1_loss(actual_reward, expected_reward)
            self.data_points.append(loss.data[0])
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            self.model.reset()
            if (check_progress and (i % ModelManager.num_progress == 0)):
                print(action_list)
                print("Huber Loss: %f" % loss.data[0])
                print("Elapsed Time: %f sec" % (time.time() - self.start_time))
                print("Time Remaining: %f sec\n" % (((time.time() - self.start_time) / (i+1)) * (ModelManager.num_epochs - (i+1))))
            if (loss.data[0] < ModelManager.gradient_threshold): break


    def test(self):
        """
        Calling this method will perform a single epoch of testing across the dataset,
        and will return the action list output by the LSTM model.
        """
        action_list = torch.stack([
            self.argmax_action(self.inputs[j][0]) for j in range(ModelManager.time_steps)
        ])
        self.model.reset()
        return action_list


    def plot(self):
        """
        This function will generate a graphical plot
        of Training Error versus Training Epoch using the matplotlib package.
        """
        plt.plot(self.data_points)
        plt.ylabel('Huber Loss')
        plt.xlabel('Training Epoch')
        plt.savefig(datetime.datetime.now().strftime("%Y:%B:%d_%H:%M") + "_huber_vs_epoch.png")


    def argmax_action(self, state):
        """
        Given the current STATE, return an action either randomly, or according
        to the model using an epsilon probability.

        state -- A tensor representing certain measurements from the environment.
        return -- A tensor representing actions to be made by the agent.
        """
        list_of_actions = torch.eye(ModelManager.action_size)
        best_action = list_of_actions[0, :]
        best_reward = self.model(torch.cat([state, best_action]).unsqueeze(0))
        for i in range(1, ModelManager.action_size):
            action = list_of_actions[i, :]
            reward = self.model(torch.cat([state, action]))
            if reward.data[0][0] > best_reward.data[0][0]:
                best_action = action
                best_reward = reward
        return best_action


    def max_action(self, state):
        """
        Given the current STATE, return the highest possible reward according
        to the model.

        state -- A tensor representing certain measurements from the environment.
        return -- A tensor representing a reward to be gained by the agent.
        """
        list_of_actions = torch.eye(ModelManager.action_size)
        best_action = list_of_actions[0, :]
        best_reward = self.model(torch.cat([state, best_action]).unsqueeze(0))
        for i in range(1, ModelManager.action_size):
            action = list_of_actions[i, :]
            reward = self.model(torch.cat([state, action]))
            if reward.data[0][0] > best_reward.data[0][0]:
                best_action = action
                best_reward = reward
        return best_reward


"""

Section 2: Testing the Model Manager

"""


manager = ModelManager()
manager.train(check_progress=True)
print(manager.test())
manager.plot()