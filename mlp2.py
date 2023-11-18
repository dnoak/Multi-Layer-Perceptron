import random
import numpy as np
import time
from dataclasses import dataclass
from typing import Callable
import matplotlib.pyplot as plt

@dataclass
class Loss:
    @staticmethod
    def L1(x, y):
        return np.mean(np.abs(x - y))

    @staticmethod
    def L2(x, y):
        return np.mean((x - y) ** 2)

@dataclass
class Activation:
    @staticmethod
    def linear(z):
        return z

    @staticmethod
    def ReLU(z):
        z[z < 0] = 0
        return z

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _XeLU(x, w, b):
        # pode ser apenas a multiplicacao
        # da matriz identidade invertida
        # (0 vira 1 e 1 vira 0)
        identity = -np.identity(len(w))
        identity[identity == 0] = 1
        relu_input = np.tile(w, (len(w), 1)) * identity * x
        relu_output = Activation.ReLU(relu_input)
        xelu = np.prod(relu_output, axis=1).sum()
        xelu *= - np.sum(np.abs(w))/np.prod(w)
        return (xelu + b).squeeze()
    
    def XeLU(x, w, b):
        signs = np.sign(x * w + b)
        equal = np.all(signs == signs[0])
        if not equal:
            return np.sum(np.multiply(x, w)) + b
        return 0

@dataclass
class Neuron:
    input_size: int
    activation: Callable
    dtype = np.float32
    
    def __post_init__(self):
        self.w = 2*(np.random.random(self.input_size).astype(self.dtype) - 1/2)
        self.w_grad = np.zeros(self.w.shape).astype(self.dtype)
        self.b = 2*(np.random.random(1).astype(self.dtype) - 1/2)
        self.b_grad = np.zeros(1).astype(self.dtype)

    def apply_grad(self):
        self.w -= self.w_grad
        self.b -= self.b_grad
        self.w_grad *= 0
        self.b_grad *= 0

    def Z(self, x):
        if x.shape != self.w.shape:
            raise ValueError('Shape: element wise multiplication:', x, self.w)
        z = np.sum(np.multiply(x, self.w)) + self.b
        return z.squeeze()
  
    def output(self, x):
        return self.activation(self.Z(x))

@dataclass
class Layer:
    height: int
    prev_height: int
    activation: Callable

    def __post_init__(self):
        self.neurons = [
            Neuron(self.prev_height, self.activation) 
            for n in range(self.height)
        ]
    
    def output(self, x):
        return np.array([n.output(x) for n in self.neurons])

@dataclass
class NeuralNetwork:
    layers: list
    activations: list
    lr: float = 10e-3
    h: float = 10e-6
    loss: Callable = Loss.L2
    random_seed: int = 3060 #np.random.randint(0, 10000)

    def __post_init__(self):
        print(f"Random seed: {self.random_seed}")
        if len(self.activations) != len(self.layers) - 1:
            raise ValueError(f"Len: activations != layers")
        self.layers = [
            Layer(h, p, a) for h, p, a in 
            zip(self.layers[1:], self.layers[:-1], self.activations)
        ]
    
    def print(self):
        print(f"layer 0: {self.layers[0].prev_height} neuros\n - ")
        for i, l in enumerate(self.layers):
            print(f"layer {i+1}: {len(l.neurons)} neurons")
            for n in l.neurons:
                print(f"{3*' '}w={len(n.w)}, b=1", end=' ')
            print()

    def apply_grads(self):
        for l in self.layers:
            for n in l.neurons:
                n.apply_grad()

    def backward(self, x, y, train_size):
        for l in self.layers:
            for n in l.neurons:
                for i in range(n.input_size):
                    n.w[i] += self.h
                    loss_w_h = self.loss(self.forward(x), y)
                    n.w[i] -= self.h
                    loss_w = self.loss(self.forward(x), y)
                    n.w_grad[i] += (loss_w_h - loss_w) / self.h * self.lr / train_size
                n.b += self.h
                loss_b_h = self.loss(self.forward(x), y)
                n.b -= self.h
                loss_b = self.loss(self.forward(x), y)
                n.b_grad += ((loss_b_h - loss_b) / self.h) * self.lr / train_size
    
    def forward(self, x):
        for l in self.layers:
            x = l.output(x)
        return x
    
    def test(self, inputs, outputs):
        for x, y in zip(inputs, outputs):
            print(f"Input: {x}, Pred: {np.round(self.forward(x), 2)}, Real: {y}") 
    
    def val_loss(self, inputs, outputs, epoch):
        loss = 0
        for x, y in zip(inputs, outputs):
            loss += self.loss(self.forward(x), y)
        print(f"Epoch:{epoch}, Loss: {loss}") 
    
    def train(self, inputs, outputs, epochs):
        train_size = len(inputs)
        for epoch in range(epochs):
            inputs_outputs = list(zip(inputs, outputs))
            random.shuffle(inputs_outputs)
            inputs, outputs = zip(*inputs_outputs)
            for x, y in zip(inputs, outputs):
                self.backward(x, y, train_size)
            self.apply_grads()
            self.val_loss(inputs, outputs, epoch)
        self.test(inputs, outputs)

@dataclass
class TrainData:
    @staticmethod
    def xor(show=False):
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([[0], [1], [1], [0]])
        return {'inputs': inputs, 'outputs': outputs}
    
    @staticmethod
    def linear(show=False):
        inputs = np.expand_dims(np.arange(10), axis=1)
        outputs = 2 * np.expand_dims(np.arange(10), axis=1) - 10
        return {'inputs': inputs, 'outputs': outputs}

    @staticmethod
    def circle(show=False):
        data_circle_xy = 15*(np.random.random((500, 2))-1/2)
        outputs = np.array([[x**2 + y**2 < 25] for x, y in data_circle_xy]).astype(np.float32)
        if show:
            red = data_circle_xy[outputs.squeeze() == 1]
            blue = data_circle_xy[outputs.squeeze() == 0]
            plt.scatter(red[:, 0], red[:, 1], c='red')
            plt.scatter(blue[:, 0], blue[:, 1], c='blue')
            plt.show()
        return {'inputs': data_circle_xy, 'outputs': outputs}

train_data = TrainData.linear(show=True)

nn = NeuralNetwork(
    layers=[1, 2, 1],
    activations=[Activation.linear, Activation.linear]
)

nn.train(**train_data, epochs=500)
