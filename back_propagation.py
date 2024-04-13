import numpy as np
from dataclasses import dataclass

@dataclass
class LossFunction: ...

@dataclass
class ActivationFunction: ...

@dataclass
class L2(LossFunction):
    @staticmethod
    def output(x, y):
        assert x.shape == y.shape, f'shape error: {x.shape} != {y.shape}'
        return np.mean((x - y) ** 2)
    
    @staticmethod
    def derivative(x, y):
        assert x.shape == y.shape, f'shape error: {x.shape} != {y.shape}'
        return 2 * (x - y)
    
@dataclass
class Linear(ActivationFunction):
    @staticmethod
    def output(z):
        return z
    
    @staticmethod
    def derivative(z):
        return 1

@dataclass
class ReLU(ActivationFunction):
    @staticmethod
    def output(z):
        z = z.copy()
        z[z < 0] = 0
        return z
    
    @staticmethod
    def derivative(z):
        return np.where(z > 0, 1, 0)

@dataclass
class Neuron:
    input_size: int
    activation: ActivationFunction
    dtype = np.float32
    input_neurons = []

    def __post_init__(self):
        self.w = self.w = 2*(np.random.random(self.input_size).astype(self.dtype) - 1/2)
        self.b = self.b = 2*(np.random.random(1).astype(self.dtype) - 1/2)
        self.w_grad = np.zeros_like(self.w)
        self.b_grad = np.zeros_like(self.b)

    def dzdw(self):
        return self.x

    def dadz(self):
        return np.expand_dims(self.activation.derivative(self.z), axis=0)
    
    def dadw(self):
        return self.dadz_derivative() * self.dzdw_derivative() 

    def output(self, x):
        assert x.shape == self.w.shape, f'shape error: {x.shape} != {self.w.shape}'
        self.x = x
        self.z = (self.x.dot(self.w) + self.b).squeeze()
        self.a = self.activation.output(self.z)
        return self.a
    
@dataclass
class Layer:
    prev_height: int
    height: int
    activation: ActivationFunction

    def __post_init__(self):
        self.neurons = [
            Neuron(self.prev_height, self.activation) 
            for n in range(self.height)
        ]

    def output(self, x):
        return np.array([neuron.output(x) for neuron in self.neurons])

@dataclass
class Layer:
    prev_height: int
    height: int
    activation: ActivationFunction

    def __post_init__(self):
        self.neurons = [
            Neuron(self.prev_height, self.activation) 
            for n in range(self.height)
        ]

    def output(self, x):
        return np.array([neuron.output(x) for neuron in self.neurons])


@dataclass
class NeuralNetwork:
    layers: list[Layer]
    lr: float
    h: float
    loss_fn: LossFunction
    random_seed: int = 1010

    def __post_init__(self):
        print(f"random seed: {self.random_seed}\n")
        np.random.seed(self.random_seed)
        self.generate_dag()
    
    def generate_dag(self):
        for l1, l2 in zip(self.layers, self.layers[1:]):
            for n2 in l2.neurons:
                n2.input_neurons = l1.neurons

    def arquiteture(self):
        print(f"{'-'*50}\n(0) Layer: inputs")
        print(*[f'|x{i}|' for i in range(self.layers[0].prev_height)])
        for i, layer in enumerate(self.layers):
            print(f"({i+1}) Layer: {len(layer.neurons)} neuron(s)")
            for neuron in layer.neurons:
                print(f"|w={len(neuron.w)} b=1|", end='  ')
            print()
        print(f"{'-'*50}\n")

    def backward(self, pred, real):
        def recursive_backward(neuron: Neuron, delta):
            neuron.w_grad = neuron.w_grad + delta * neuron.dadz() * neuron.dzdw()
            neuron.b_grad = neuron.b_grad + delta * neuron.dadz() * 1
            delta = delta * neuron.dadz() * neuron.w
            for d, prev_neuron in enumerate(neuron.input_neurons):
                recursive_backward(prev_neuron, delta[d])

        dlda = self.loss_fn.derivative(pred, real)
        for n, neuron in enumerate(self.layers[-1].neurons):
            recursive_backward(neuron, dlda[n] / len(dlda))

    def backward_limit(self, x, pred, real):
        loss = L2.output(pred, real)
        for l, layer in enumerate(self.layers[::-1]):
            for n, neuron in enumerate(layer.neurons):
                print(f'--> layer {l}, neuron {n}, w:')
                #print(f'x: {neuron.x}\nw: {neuron.w}\nz: {neuron.z}\na: {neuron.a}')
                #print(f'dadz: {neuron.dadz()}\ndzdw: {neuron.dzdw()}')
                lim_w = np.zeros_like(neuron.w_grad)
                for k in range(len(neuron.w)):
                    neuron.w[k] += self.h
                    loss_h = self.loss_fn.output(self.forward(x), real)
                    neuron.w[k] -= self.h
                    lim_w[k] = ( loss_h - loss ) / self.h
                neuron.b += self.h
                loss_h = self.loss_fn.output(self.forward(x), real)
                neuron.b -= self.h
                lim_b = ( loss_h - loss ) / self.h
                print('wg/limw:', neuron.w_grad/lim_w)
                print('bg/limb:', neuron.b_grad/lim_b)
                input()

    def forward(self, x):
        for l in self.layers:
            x = l.output(x)
        return x

nn = NeuralNetwork(
    layers=[
        Layer(2, 3, ReLU()),
        Layer(3, 5, ReLU()),
        Layer(5, 3, Linear()),
        Layer(3, 1, Linear()),
    ],
    lr=0.01,
    h=10e-5,
    loss_fn=L2,
    random_seed=1020
)

nn.arquiteture()

x = np.array([3, 3]).astype(np.float64)
y = np.array([3]).astype(np.float64)

pred = nn.forward(x)
nn.backward(pred, y)
nn.backward_limit(x, pred, y)
