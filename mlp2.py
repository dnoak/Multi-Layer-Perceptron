import random
import numpy as np
from dataclasses import dataclass
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
        z = z.copy()
        z[z < 0] = 0
        return z

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def _XeLU(x1, x2):
        x1 = np.expand_dims(x1, axis=0)
        x2 = np.expand_dims(x2, axis=0)
        s1 =  Activation.ReLU(x1) * Activation.ReLU(-x2)
        s2 =  Activation.ReLU(-x1) * Activation.ReLU(x2)
        if x1 * x2 == 0:
            return np.zeros(1)
        return - (s1 + s2) * (np.abs(x1) + np.abs(x2)) / (x1 * x2)

    @staticmethod
    def XeLU(x, w, b):
        total_sum = b.copy()
        for i in range(len(x)):
            for j in range(i, len(x)):
                total_sum += Activation._XeLU(x[i]*w[j], x[j]*w[j])
        return (total_sum.squeeze())

@dataclass
class Neuron:
    input_size: int
    activation: callable
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
        if self.activation == Activation.XeLU:
            return self.activation(x, self.w, self.b)
        return self.activation(self.Z(x))

@dataclass
class FourierNeuron:
    input_size: np.ndarray
    dtype: np.dtype = np.float32

    def __post_init__(self):
        self.w = 2*(np.random.random(self.input_size).astype(self.dtype) - 1/2)
        self.w_grad = np.zeros(self.w.shape).astype(self.dtype)
    
    def apply_grad(self):
        self.w -= self.w_grad
        self.w_grad *= 0
    
    def Z(self, x):
        #if x.shape != self.w.shape:
            #raise ValueError(f"Shape mismatch: {x.shape} and {self.w.shape}")
        return x * self.w
    
    def output(self, x):
        return np.fft.ifft(self.Z(x)).real

@dataclass  
class FourierLayer:
    input_size: np.ndarray
    dtype: np.dtype = np.float32

    def __post_init__(self):
        self.output_size = self.input_size
        self.neurons = [FourierNeuron(self.input_size, self.dtype)]

    def output(self, x):
        return self.neurons[0].output(x)

@dataclass
class Layer:
    height: int = None
    prev_height: int = None
    activation: callable = Activation.linear

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
    lr: float = 20e-4
    h: float = 10e-6
    loss: callable = Loss.L2
    graphics = False
    random_seed: int = 1010

    def __post_init__(self):
        print(f"Random seed: {self.random_seed}")
        np.random.seed(self.random_seed)
    
    def arquiteture(self):
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

    def backward_neuron(self, n, x, y, train_size):
        loss = self.loss(self.forward(x), y)
        for i in range(n.input_size):
            n.w[i] += self.h
            loss_w_h = self.loss(self.forward(x), y)
            n.w[i] -= self.h
            n.w_grad[i] +=  ( ((loss_w_h - loss) / self.h) * self.lr / train_size )
        n.b += self.h
        loss_b_h = self.loss(self.forward(x), y)
        n.b -= self.h
        n.b_grad += ( ((loss_b_h - loss) / self.h) * self.lr / train_size )

    def backward_fourier_neuron(self, n, x, y, train_size):
        for i in range(n.input_size):
            n.w[i] += self.h
            loss_w_h = self.loss(self.forward(x), y)
            n.w[i] -= self.h
            loss_w = self.loss(self.forward(x), y)
            n.w_grad[i] += (loss_w_h - loss_w) / self.h * self.lr / train_size

    def backward(self, x, y, train_size):
        for l in self.layers:
            for n in l.neurons:
                if isinstance(n, Neuron):
                    self.backward_neuron(n, x, y, train_size)
                elif isinstance(n, FourierNeuron):
                    self.backward_fourier_neuron(n, x, y, train_size)

    def forward(self, x):
        for l in self.layers:
            x = l.output(x)
        return x
    
    def test(self, inputs, outputs):
        for x, y in zip(inputs, outputs):
            y_pred = np.round(self.forward(x), 2)
            print(f"Input: {x}, Pred: {y_pred}, Real: {y}") 

    def visual_loss(self, x, y_pred, y):
        x = x.squeeze().astype(np.float32)
        y_pred = y_pred.squeeze().astype(np.float32)
        y = y.squeeze().astype(np.float32)
        
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min([np.min(y), np.min(y_pred)]), np.max([np.max(y), np.max(y_pred)])

        norm_x = (x - min_x) / (max_x - min_x)
        norm_y = (y - min_y) / (max_y - min_y)
        norm_y_pred = (y_pred - min_y) / (max_y - min_y)

        plt.scatter(norm_x, norm_y, c='blue')
        plt.scatter(norm_x, norm_y_pred, c='red')
        plt.show()

    def train_loss(self, x, y, epoch):
        y_pred = []
        for xi in x:
            y_pred += [self.forward(xi)]
        if self.graphics:
            self.visual_loss(x, np.array(y_pred), y)
        print(f"Epoch:{epoch}, Loss: {np.sum(self.loss(y_pred, y))}") 
    
    def train(self, x, y, epochs):
        train_size = len(x)
        for epoch in range(epochs):
            x_y = list(zip(x, y))
            random.shuffle(x_y)
            x, y = zip(*x_y)
            x, y = np.array(x)[:10], np.array(y)[:10]
            
            for xi, yi in zip(x, y):
                self.backward(xi, yi, train_size)          
            self.apply_grads()
            self.train_loss(x, y, epoch)
        self.test(x, y)

@dataclass
class TrainData:
    random_seed: 1010

    def __post_init__(self):
        np.random.seed(self.random_seed)
    
    @staticmethod
    def xor(show=False):
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([[0], [1], [1], [0]])
        return {'x': inputs, 'y': outputs}
    
    @staticmethod
    def linear(show=False):
        inputs = np.expand_dims(np.arange(10), axis=1)
        outputs = 2 * np.expand_dims(np.arange(10), axis=1) - 10
        return {'x': inputs, 'y': outputs}
    
    @staticmethod
    def xsquared(show=False):
        inputs = np.expand_dims(100*(np.random.random(100)), axis=1)
        outputs = inputs ** 2
        return {'x': inputs, 'y': outputs}
 
    @staticmethod
    def circle(samples, r, show=False):
        data_circle_xy = (np.sqrt(2*np.pi*r**2))*(np.random.random((samples, 2))-1/2)
        outputs = np.array([[1 if x**2 + y**2 < r**2 else 0] for x, y in data_circle_xy]).astype(np.float32)
        if show:
            print(outputs[outputs == 1].shape, outputs[outputs == 0].shape)
            red = data_circle_xy[outputs.squeeze() == 1]
            blue = data_circle_xy[outputs.squeeze() == 0]
            plt.scatter(red[:, 0], red[:, 1], c='red')
            plt.scatter(blue[:, 0], blue[:, 1], c='blue')
            plt.show()
        return {'x': data_circle_xy, 'y': outputs}
    
    @staticmethod
    def sin(show=False):
        size=40
        x = np.arange(size)
        inputs = np.zeros((100, size))
        outputs = np.zeros((100, size))
        for i in range(100):
            phase = np.random.randint(0, 4)
            amp = np.random.random()*10
            outputs[i] = amp*np.sin(x*np.pi/2 + phase)
            inputs[i] = outputs[i].copy()
            for n in range(5):
                noise_phase = (np.random.random())*np.pi
                noise_amp = (np.random.random())*amp
                inputs[i] += (amp+noise_amp)*(np.sin(x + (phase+noise_phase))) * (n+2)**-1.5

        inputs = np.expand_dims(inputs, axis=2)
        outputs = np.expand_dims(outputs, axis=2)
        if show:
            for i in range(size):
                plt.plot(np.arange(size), inputs[i], c='blue')
                plt.plot(np.arange(size), outputs[i], c='red')
                plt.show()
        
        return {'x': inputs, 'y': outputs}

train_data = TrainData.circle(100, 10, show=False)

nn = NeuralNetwork(
    layers=[
        Layer(prev_height=1, height=2, activation=Activation.ReLU),
        Layer(prev_height=2, height=2, activation=Activation.ReLU),
        Layer(prev_height=2, height=2, activation=Activation.ReLU),
        Layer(prev_height=2, height=1, activation=Activation.linear),
    ],
    lr=0.03,
    loss=Loss.L2,
    random_seed=np.random.randint(0, 10000)
)

for x in train_data['x'][:10]:
    print(x, nn.forward(x))

nn.train(**train_data, epochs=200)
