from matplotlib import pyplot as plt
import numpy as np
from dataclasses import dataclass

@dataclass
class Loss:
    @staticmethod
    def L1(x, y):
        return np.mean(np.abs(x - y))

    @staticmethod
    def L2(x, y):
        return np.mean((x - y) ** 2)

@dataclass
class FourierNeuron:
    input_size: np.ndarray
    activation: callable = None
    dtype: np.dtype = np.float32

    def __post_init__(self):
        self.w = 20*(np.random.random(self.input_size).astype(self.dtype) - 1/2)
        self.w_grad = np.zeros(self.w.shape).astype(self.dtype)

    def apply_grad(self):
        self.w -= self.w_grad
        self.w_grad *= 0

    def Z(self, x):
        return x.squeeze() * self.w

    def forward(self, x):
        return np.fft.ifftshift(self.Z(x))


@dataclass
class PeriodicTrainData:
    random_seed: int = np.random.randint(0, 1000)

    def __post_init__(self):
        np.random.seed(self.random_seed)
    
    @staticmethod
    def random_A_phi_t():
        A = 8 * (np.random.random() + 1/2) + 4
        phi = np.random.random() * np.pi/2
        t = np.random.randint(1, 11)
        return A, phi, t
    
    @staticmethod
    def random_noise(max_size):
        return 2*(np.random.random()-1/2) * max_size

    @staticmethod
    def x_times_cosx(samples, show=False):
        A, phi, t = PeriodicTrainData.random_A_phi_t()
        f = lambda x: A * np.cos(x * np.pi/(1+t) + phi) ** 2
        x_train = np.array([[x * f(x) + PeriodicTrainData.random_noise(A/2)] for x in range(samples)])
        y_train = np.array([[f(y)] for y in range(samples)])
        if show:
            plt.plot(np.arange(samples), x_train, c='blue')
            plt.plot(np.arange(samples), y_train, c='red')
            plt.show()
        return np.round(x_train, 2), np.round(y_train, 2)
    
    @staticmethod
    def cosx_plus_random(samples, show=False):
        A, phi, t = PeriodicTrainData.random_A_phi_t()
        f = lambda x: A * np.cos(x * np.pi/t + phi)
        x_train = np.array([[f(x) + PeriodicTrainData.random_noise(A/2)] for x in range(samples)])
        y_train = np.array([[f(y)] for y in range(samples)])
        if show:
            plt.plot(np.arange(samples), x_train, c='blue')
            plt.plot(np.arange(samples), y_train, c='red')
            plt.show()
        return np.round(x_train, 2), np.round(y_train, 2)

samples = 100
x_train, y_train = PeriodicTrainData.cosx_plus_random(samples, show=False)


fn = FourierNeuron((samples))
h = 0.0001
lr = 0.1

first_forward = fn.forward(x_train).copy()
#print(first_forward.shape)


x, y = x_train, y_train

for epoch in range(500):
    prev_loss = Loss.L2(fn.forward(x), y)
    for n in range(len(fn.w)):
        fn.w[n] += h
        loss_w_h = Loss.L2(fn.forward(x), y)
        fn.w[n] -= h
        loss_w = Loss.L2(fn.forward(x), y)
        fn.w_grad[n] = (loss_w_h - loss_w) / h * lr
    fn.apply_grad()
    loss = Loss.L2(fn.forward(x), y)

    print(f"epoch={epoch}, loss={loss}, lr={lr}")
    # if loss > prev_loss:
    #     break

#plt.plot(np.arange(samples), x_train, c='blue')
plt.plot(np.arange(samples), y_train, c='green')
plt.plot(np.arange(samples), fn.forward(x_train), c='red')
#plt.plot(np.arange(samples), first_forward, c='orange')
plt.show()

