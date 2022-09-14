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

import pickle
import numpy as np

'''Created a parent class named Layer which defined the forward and backward functions that are used by all layers.
In this way, we can take advantage of polymorphism to easily compute the forward and backward passes of the entire network.'''

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError


# inherit from base class Layer
class LinearLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
  
    def weights(self):
      print(self.weights)

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_loss, learning_rate):
        input_loss = np.dot(output_loss, self.weights.T)
        weights_error = np.dot(self.input.T, output_loss)
        # dBias = output_error

        # update weights and bias
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_loss
        return input_loss


class ActivationLayer(Layer):
    def __init__(self, activation, activation_grad):
        self.activation = activation
        self.activation_grad = activation_grad

    # returns the activated input
    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_loss, learning_rate):
        return self.activation_grad(self.input) * output_loss


def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_gradient(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;


#Sigmoid Function
class Sigmoid(Layer):
    def __init__(self):
        self.outputs = None

    def forward(self, inputs):
        self.outputs = 1. / (1. + np.exp(-inputs))
        return self.outputs

    def backward(self, gradient_input):
        return self.outputs  * (1 - self.outputs) * gradient_input

    
#Hyperbolic Tangent Function
class Tanh(Layer):
    def __init__(self):
        self.outputs = None

    def forward(self, inputs):
        return np.tanh(inputs);

    def backward(self, gradient_input):
        return 1-np.tanh(gradient_input)**2;

#Softmax Function
class Softmax(Layer):
    def __init__(self):
        self.N = None
        self.dims = None
        self.outputs = None

    def forward(self, inputs):
        self.N = np.exp(inputs - np.max(inputs))
        return self.N / np.sum(self.N, axis=0, keepdims=True)

    def backward(self, probs, errs):
        self.dims = probs.shape[1]
        self.outputs = np.empty(probs.shape)
        for dim in range(self.dims):
            dim_xy = -(probs * probs[:,[dim]])
            dim_xy[:,dim] += probs[:,dim]
            self.outputs[:,dim] = np.sum(errs * dim_xy, axis=1)
        return self.outputs

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

class crEntropy(Layer):
    def __init__(self):
        self.output = None
        
    def forward(self, pred, target):
        self.output =  -target * np.log(pred)
        return self.output
            
    def backward(self,pred, target):
        return target - pred

# class CustomStopper(EarlyStopping):
#     def __init__(self, monitor='val_loss',
#              min_delta=0, patience=0, verbose=0, mode='auto', start_epoch = 100): # add argument for starting epoch
#         super(CustomStopper, self).__init__()
#         self.start_epoch = start_epoch

#     def on_epoch_end(self, epoch, logs=None):
#         if epoch > self.start_epoch:
#             super().on_epoch_end(epoch, logs)

#Sequential Class
class Sequential(Layer):
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_grad = None
        self.weights = None

    def add(self, layer):
        self.layers.append(layer)

    def use_loss(self, loss, loss_grad):
        self.loss = loss
        self.loss_grad = loss_grad

    def predict(self, input):
        samples = len(input)
        result = []
    
        for i in range(samples):
            # forward propagation
            output = input[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)
                # computing loss
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_grad(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            # calculating error on all the samples
            err /= samples
            if i % 5 == 0:
              print('epoch %d/%d   error=%f' % (i+1, epochs, err))

    def getWeights(self):
      self.weights = list()
      for layer in self.layers:
        if isinstance(layer, LinearLayer):
          self.weights.append(layer.weights)
          return self.weights


    def saveModel(self, file="NeuralNetwork.weights"):
      f = open(file, mode="wb")
      pickle.dump(self.getWeights(), f)
      f.close()


    def loadModel(self, file="Neural_Network.weights"):
        f = open(file, mode="rb")
        weights = pickle.load(f)
        f.close()
        return weights
