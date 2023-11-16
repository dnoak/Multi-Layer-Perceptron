import numpy as np
from dataclasses import dataclass
import time

@dataclass
class LossFunctions:
    @staticmethod
    def r1(x, y):
        return np.mean(np.abs(x - y))
    
    @staticmethod
    def r2(x, y):
        return np.mean((x - y) ** 2)

@dataclass
class ActivationFunctions:
    @staticmethod
    def linear(z):
        return z
    
    @staticmethod
    def ReLU(z):
        return max(0, z)
    
    @staticmethod
    def tanh(z):
        return np.tanh(z)
    
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

@dataclass
class Neuron(ActivationFunctions):
    input_size: int
    dtype = np.float64
    def __post_init__(self):
        self.activation_fn = self.ReLU
        self.w = 2*(np.random.random(self.input_size).astype(self.dtype)-1/2)
        self.wg = np.zeros(self.w.shape, dtype=self.dtype)
        self.b = 2*(np.random.random(1).astype(self.dtype)-1/2)
        self.bg = np.zeros(self.b.shape, dtype=self.dtype)
        
    def apply_grad(self):
        self.w -= self.wg
        self.wg = np.zeros(self.w.shape, dtype=self.dtype)
        self.b -= self.bg
        self.bg = np.zeros(self.b.shape, dtype=self.dtype)

    def Z(self, x):
        z = np.sum(x * self.w) + self.b
        return z.squeeze()
    
    def output(self, x):
        return self.activation_fn(self.Z(x))
        
@dataclass
class Layer:
    depth: int
    prev_depth: int
    def __post_init__(self):
        self.neurons = [Neuron(self.prev_depth) for n in range(self.depth)]
        self.neurons[-1].activation_fn = ActivationFunctions.linear
    
    def output(self, x):
        x = np.array([n.output(x) for n in self.neurons])
        return x

@dataclass
class NeuralNetwork:
    arquiteture: list[dict]
    lr: float = 0.01
    h: float = 0.000001
    loss_fn: id = LossFunctions.r2
    random_seed: int = np.random.randint(0, 1000)
    def __post_init__(self):
        np.random.seed(self.random_seed)
        ins_outs = [(l1, l0) for l1, l0 in zip(self.arquiteture[1:], self.arquiteture[:-1])]
        self.layers = [Layer(*io) for io in ins_outs]
    
    def print(self):
        print(self)
        for l in self.layers:
            print(' '*2, l)
            for n in l.neurons:
                print(' '*4, n, '\n', ' '*6, n.w, n.b)
        
    def apply_grads(self):
        for l in self.layers:
            for n in l.neurons:
                n.apply_grad()
    
    def grad(self, y2, y1):
        slope = (y2 - y1) / self.h
        return (slope) * self.lr

    def backward(self, x, y, train_size):
        for l in self.layers:
            for n in l.neurons:
                for wi in range(n.input_size):
                    n.w[wi] += self.h
                    fw2 = self.loss_fn(self.forward(x), y)
                    n.w[wi] -= self.h
                    fw1 = self.loss_fn(self.forward(x), y)
                    n.wg[wi] += self.grad(fw2, fw1)
                n.b += self.h
                fb2 = self.loss_fn(self.forward(x), y)
                n.b -= self.h
                fb1 = self.loss_fn(self.forward(x), y)
                n.bg += self.grad(fb2, fb1)
        self.apply_grads()
        
    def forward(self, x):
        for l in self.layers:
            x = l.output(x)
        return x
    
    def train(self, train_data, epochs):
        train_size = len(train_data['input'])
        for epoch in range(epochs):
            loss = 0
            for x, y in zip(*train_data.values()):                        
                self.backward(x, y, train_size)
                loss += self.loss_fn(self.forward(x), y)
            if epoch % 10 == 0:
                print('Epoch =', epoch, f", Loss: {loss/train_size}")

        print('Resultados: ')
        for x, y in zip(*train_data.values()):
            print(x, self.forward(x))

        
train_data_XOR = {
    'input': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    'output': np.array([[0], [1], [1], [0]])
}

train_data_linear = {
    'input':  -np.expand_dims(np.arange(10), axis=1),
    'output': -np.expand_dims(2 * np.arange(10) + 10, axis=1)
}

nn = NeuralNetwork(
    arquiteture=[2, 5, 1]
)

nn.train(train_data_XOR, 500)
#nn.backward(1, 2)


