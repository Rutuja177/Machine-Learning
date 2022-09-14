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

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

#using Tanh activation function
net = Sequential()
net.add(LinearLayer(2, 2))
tan = Tanh()
net.add(ActivationLayer(tan.forward, tan.backward))
net.add(LinearLayer(2, 1))
net.add(ActivationLayer(tan.forward, tan.backward))


net.use_loss(mse, mse_gradient)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

output = net.predict(x_train)
print("Predict:", output)

net.getWeights()

net.saveModel()
XOR_solved_weights = net.loadModel()
print("weights:", XOR_solved_weights)


#Using Sigmoid activation function
net = Sequential()
net.add(LinearLayer(2, 2))
sigmoid = Sigmoid()
net.add(ActivationLayer(sigmoid.forward, sigmoid.backward))
net.add(LinearLayer(2, 1))
net.add(ActivationLayer(sigmoid.forward, sigmoid.backward))

net.use_loss(mse, mse_gradient)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

output2 = net.predict(x_train)
print("Predict:", output2)

net.getWeights()

net.saveModel()
XOR_solved_weights = net.loadModel()
print("weights:", XOR_solved_weights)




