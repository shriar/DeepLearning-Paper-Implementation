
from tinygrad import Tensor, nn, TinyJit, GlobalCounters
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt
from IPython import display
import torch


class Controller:
    def __init__(self, num_inputs, controller_out_dim, controller_hid_dim):
        # print("--- Initialize Controller")
        self.fc1 = nn.Linear(num_inputs, controller_hid_dim)
        self.fc2 = nn.Linear(controller_hid_dim, controller_out_dim)
    def __call__(self, x, last_read):
        # print("x", x.shape, '\tlast_read: ', last_read.shape)
        x = Tensor.cat(x, last_read, dim=1)
        # print("x", x.shape)
        x = Tensor.sigmoid(self.fc1(x))
        x = Tensor.sigmoid(self.fc2(x))
        return x

def _convolve(w, s):
    b, d = s.shape
    assert b == 1
    assert d == 3

    w = Tensor.squeeze(w)
    t = Tensor.cat(w[-1:], w, w[:1])
    c = Tensor.conv2d(t.view(1, 1, -1), s.view(1, 1, -1))
    return c.view(b, -1)

_convolve(Tensor.rand(1, 5), Tensor.rand(1, 3)).shape


class Memory():
    def __init__(self, M, N):
        self.M = M # numer of memory locations
        self.N = N # vector size at each location
        self.read_lengths = self.N + 1 + 1 + 3 + 1
        self.write_lengths = self.N + 1 + 1 + 3 + 1 + self.N + self.N
        self.w_last = []
        self.reset_memory()

    def address(self, k, β, g, s, γ, memory, w_last):
        # Content focus
        wc = self._similarity(k, β, memory)

        # Location focus
        wg = self._interpolate(wc, g, w_last)
        # print(wg.shape, s.shape)
        ŵ = self._shift(wg, s)
        w = self._sharpen(ŵ, γ)
        
        return w
    
    def reset_memory(self):
        self.w_last = []
        self.w_last.append(Tensor.zeros(1, self.M, requires_grad=True))

    def _similarity(self, k, β, memory):
        dot_product = Tensor.sum(k * memory, axis=-1)

        magnitude_k = Tensor.sqrt(Tensor.sum(k ** 2, axis=-1))
        magnitude_memory = Tensor.sqrt(Tensor.sum(memory ** 2, axis=-1))

        epsilon = 1e-7
        denominator = magnitude_k * magnitude_memory + epsilon

        cosine_similarity = dot_product / denominator
        _similarity = Tensor.softmax(β*cosine_similarity, axis=-1)
        return _similarity
    
    def _interpolate(self, wc, g, w_last):
        return g * wc + (1 - g) * w_last

    def _shift(self, wg, s):
        return _convolve(wg, s)
    
    def _sharpen(self, ŵ, γ):  
        w = ŵ ** γ
        w = Tensor.div(w, Tensor.sum(w, axis=-1))
        return w


class ReadHead(Memory):
    def __init__(self, M, N, controller_out_dim):
        super().__init__(M, N)
        print("--- Initialize Memory: ReadHead")
        self.fc_read = nn.Linear(controller_out_dim, self.read_lengths)

    def read(self, memory, w):
        return w @ memory
    
    def __call__(self, controller_out, memory): 
        param = self.fc_read(controller_out)
        k, β, g, s, γ = Tensor.split(param.squeeze(), [self.N, 1, 1, 3, 1])

        k = Tensor.tanh(k)
        β = Tensor.softplus(β)
        g = Tensor.sigmoid(g)
        s = Tensor.softmax(s, axis=0).view(1, -1)
        γ = 1 + Tensor.softplus(γ)

        w = self.address(k, β, g, s, γ, memory, self.w_last[-1])
        self.w_last.append(w)
        read = self.read(memory, w)
        return read
    

class WriteHead(Memory):
    def __init__(self, M, N, controller_out_dim):
        super().__init__(M, N)
        
        print("--- Initialize Memory: WriteHead")
        self.fc_write = nn.Linear(controller_out_dim, self.write_lengths)

    def write(self, memory, w, e, a):
        w = Tensor.squeeze(w)
        e = Tensor.squeeze(e)
        a = Tensor.squeeze(a)

        erase = w.view(-1, 1) @ e.view(1, -1)
        add = w.view(-1, 1) @ a.view(1, -1)

        m_tilde = memory * (1 - erase)
        updated_memory = m_tilde + add
        return updated_memory
    
    def __call__(self, controller_out, memory):
        param = self.fc_write(controller_out)

        k, β, g, s, γ, a, e = Tensor.split(param, [self.N, 1, 1, 3, 1, self.N, self.N], dim=1)

        k = Tensor.tanh(k)
        β = Tensor.softplus(β)
        g = Tensor.sigmoid(g)
        s = Tensor.softmax(s, axis=-1)
        γ = 1 + Tensor.softplus(γ)
        a = Tensor.tanh(a)
        e = Tensor.sigmoid(e)

        w = self.address(k, β, g, s, γ, memory, self.w_last[-1])
        self.w_last.append(w)
        mem = self.write(memory, w, e, a)
        return mem

class NTM():
    def __init__(self, M, N, num_inputs, num_outputs, controller_out_dim, controller_hid_dim):
        
        print("----------- Build Neural Turing machine -----------")
        self.M = M
        self.N = N
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        self.memory = Tensor.zeros(self.M, self.N, requires_grad=True)
        self.last_read = Tensor.zeros(1, self.N, requires_grad=True)

        self.controller = Controller(self.num_inputs + self.N, controller_out_dim, controller_hid_dim)
        # self.controller = Controller(self.N, controller_out_dim, controller_hid_dim)
        self.readhead = ReadHead(self.M, self.N, controller_out_dim)
        self.writehead = WriteHead(self.M, self.N, controller_out_dim)

        self.fc_out = nn.Linear(self.num_inputs + N, self.num_outputs)

    def _read_write(self, controller_out):
        # READ
        read = self.readhead(controller_out, self.memory)
        self.last_read = read

        # WRITE
        mem = self.writehead(controller_out, self.memory)
        self.memory = mem

    def __call__(self, X=None):
        if X is None:
            X = Tensor.rand(1, self.num_inputs)
        
        controller_out = self.controller(X, self.last_read)
        self._read_write(controller_out)

        # OUTPUT
        out = Tensor.cat(X, self.last_read, dim=-1)
        out = Tensor.sigmoid(self.fc_out(out))
        return out
    

class BinaySeqDataset(Dataset):

    def __init__(self, args):
        self.seq_len = args['sequence_length']
        self.seq_width = args['token_size']
        self.dataset_dim = args['training_samples']

    def _generate_seq(self):
        seq = np.random.binomial(1, 0.5, (self.seq_len, self.seq_width))
        seq = torch.from_numpy(seq)
        # Add start and end token
        inp = torch.zeros(self.seq_len + 2, self.seq_width)
        inp[1:self.seq_len + 1, :self.seq_width] = seq.clone()
        inp[0, 0] = 1.0
        inp[self.seq_len + 1, self.seq_width - 1] = 1.0
        outp = seq.data.clone()

        return inp.float(), outp.float()

    def __len__(self):
        return self.dataset_dim

    def __getitem__(self, idx):
        inp, out = self._generate_seq()
        return inp, out

args = {
    "sequence_length": 3,
    "token_size": 10,
    "training_samples": 999999,
    "memory_capacity": 64,
    "memory_vector_size": 128,
    "controller_output_dim": 256,
    "controller_hidden_dim": 512
    
    
    
}
dataset = BinaySeqDataset(args)
train_loader = DataLoader(dataset, batch_size=1,
                        shuffle=True, 
)


model = NTM(M=args["memory_capacity"],
            N=args["memory_vector_size"],
            num_inputs=args["sequence_length"]+2,
            num_outputs=args["sequence_length"],
            controller_out_dim=args["controller_output_dim"],
            controller_hid_dim=args["controller_hidden_dim"],
            )



train_losses = []
val_losses = []
total_num_epochs = 0

inp_seq_len = args["sequence_length"] + 2
out_seq_len = args["sequence_length"]


learning_rate = 1e-4
optimizer = nn.optim.Adam(nn.state.get_parameters(model))
num_epochs = 10


for epoch in tqdm.tqdm(range(num_epochs)):
    total_num_epochs += epoch
    total_train_loss = 0
    
    for batch in train_loader:
        with Tensor.train():
            x, y = Tensor(batch[0].numpy(), requires_grad=True).squeeze(), Tensor(batch[1].numpy(), requires_grad=True).squeeze()
            optimizer.zero_grad()
            for t in range(0, inp_seq_len):
                model(x[:, t].view(1, -1))

            y_pred = Tensor.zeros(y.shape, requires_grad=True).contiguous() + Tensor(1e-16, requires_grad=True)
            for i in range(0, out_seq_len):
                v = Tensor(model().numpy())
                y_pred[:, i] = v[0]


            loss = y_pred.binary_crossentropy(y)
            loss = loss.backward()
            

            optimizer.step()
            total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    

    
    display.clear_output(wait=True)
    print(f"Epoch: {epoch+1} - Train Loss: {avg_train_loss:.3f}")
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.show(block=False)


