'''
UTA ID: 1001730748
Rutuja Dukhande
CSE 6363: ML asiignment 2 neural networks
References
https://github.com/ajdillhoff/CSE6363/blob/main/neural_networks/mlp.ipynb
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
https://www.geeksforgeeks.org/deep-neural-network-with-l-layers/
'''

from NuralNetwork import *
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

def get_one_hot(y, samples):
    return np.eye(samples)[np.array(y).reshape(-1)].reshape(list(y.shape) + [samples])

softmax = Softmax()
cross_entropy = crEntropy()

class NN_withMnist:
    def __init__(self, low=-10, high=10, train_size=60000, dataset_size=70000):
        self.inputs, self.outputs = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        
        #Normalization
        self.inputs = self.inputs/255.0
        features= self.inputs.shape[1]
        n = 10

        self.outputsWeights = np.random.uniform(low=low, high=high, size=(n, features + 1)) - 0.5

        opt1 = self.outputsWeights@np.concatenate((np.zeros((self.inputs.shape[0], 1)), self.inputs), axis=1).T
        opt2 = softmax.forward(opt1)

        self.loss = cross_entropy.forward(opt2.T, get_one_hot(self.outputs.astype(int), 10))
        
    def train(self, batch=128, lr=1e-3, epochs=10000):
        self.losses = list()
        for i in range(0, self.inputs.shape[0], batch):
            f_idx, l_idx = i, self.inputs.shape[0]
            if ((f_idx + batch) < l_idx):
                l_idx = f_idx + batch

            b_ins, b_outs = self.inputs[f_idx:l_idx], get_one_hot(self.outputs[f_idx:l_idx].astype(int), 10)

            opt1 = self.outputsWeights@np.concatenate((np.zeros((b_ins.shape[0], 1)), b_ins), axis=1).T
            opt2 = softmax.forward(opt1)

            loss = cross_entropy.forward(opt2.T, b_outs)
            self.losses.append(np.sum(np.mean(loss, axis=0)))
        
            derivative_loss = cross_entropy.backward(opt2.T, b_outs)
            derivative_output_weights = derivative_loss.T@np.concatenate((np.zeros((b_ins.shape[0], 1)), b_ins), axis=1)

            self.outputsWeights = self.outputsWeights + lr * derivative_output_weights
        
    def plot_curve(self, title):
        plt.plot(self.losses)
        plt.title(title)

        plt.draw()
        plt.show()


#Initialize the model with all zeros.
model_1 = NN_withMnist(low=0, high=0)

model_1.train()
model_1.plot_curve(title="With all zeros")

#Initialize the model with random values between -10 and 10.

model_2 = NN_withMnist(low=-10, high=10)
model_2.train()
model_2.plot_curve(title="Random values betweek -10 to 10")


#Train on MNIST using a learning rate of 1 and then again with a learning rate of
#0.001. Plot the training loss curve in both instances.

model_3a = NN_withMnist()
model_3a.train(lr=1, batch=16)
model_3a.plot_curve(title="Learning Rate is 1")

model_3b = NN_withMnist()
model_3b.train(lr=0.001)
model_3b.plot_curve(title="Learning Rate is 0.001")



