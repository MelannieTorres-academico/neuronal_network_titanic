#https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
#https://blog.zhaytam.com/2018/08/15/implement-neural-network-backpropagation/
import numpy as np
import pandas as pd
import math
from statistics import stdev

# activation function
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

# backprop to adjust weights depending on the magnitude of the error
def sigmoid_derivative(x):
    return x * (1.0 - x)


def regularization(column):
    sum = 0
    for value in column:
        sum += value
    average = sum/len(column)
    std = stdev(column)
    aux_column = []
    for value in column:
        aux_column.append((value-average)/std) #https://maristie.com/blog/differences-between-normalization-standardization-and-regularization/
        #https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e
    return aux_column

def one_hot_encoding(X, column_name):
    possible_values = X[column_name].unique()
    n_possible_values = len(possible_values)
    aux_matrix = [[0]*n_possible_values]*len(X[column_name])
    df = pd.DataFrame(aux_matrix, columns= possible_values)

    for i in range(len(X[column_name])):
        value = X[column_name].iloc[i]
        df.loc[i][value] = 1
    X = X.drop(column_name, axis=1)
    X = pd.concat([X, df], axis=1, join_axes=[X.index])

    return X

class NeuralNetwork:
    def __init__(self, x, y, n_layers):
        self.input          = x
        self.y              = y
        self.learning_rate  = 0.01
        self._layers        = []
        self.weights        = []
        self.deltas         = [0]*n_layers
        self.sq_mean_error  = 0
        self.cross_entropy  = 0

    # n_inputs = The input size (coming from the input layer or a previous hidden layer)
    # n_neurons = The number of neurons in this layer.
    def add_layer(self, n_inputs, n_neurons):
        self._layers.append(None)
        weights = np.random.rand(n_inputs, n_neurons)
        self.weights.append(weights)

    def feedforward(self, row):
        current_y = self.y.iloc[row,:]['Survived']
        current_x = self.input.iloc[row,:]
        last_layer_index = len(self._layers)-1
        #refactor for scalability
        for i in range(len(self._layers)):
            if (i == 0 ): #if first layer
                self._layers[i]= sigmoid(np.dot(current_x, self.weights[i]))
            else:
                self._layers[i] = sigmoid(np.dot(self._layers[i-1], self.weights[i]))
        self.sq_mean_error = 0.5 *((current_y - self._layers[last_layer_index]) ** 2)
        self.cross_entropy  = -current_y*math.log(self._layers[last_layer_index])-(1-current_y)*math.log(1-self._layers[last_layer_index])

    def backprop(self, row):
        current_y = self.y.iloc[row,:]['Survived']
        current_x = self.input.iloc[row,:]
        last_layer_index = len(self._layers)-1
            # Loop over the layers backward
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            # If this is the output layer
            if i == last_layer_index:
                error =  current_y - layer
            else:
                error = np.dot(self.weights[i+1], self.deltas[i+1]) #get weights & deltas of next layers
            self.deltas[i] = error * sigmoid_derivative(layer)
        # Update the weights
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # The input is either the previous layers output or X itself (for the first hidden layer)
            input_to_use = np.atleast_2d(current_x if i == 0 else self._layers[i - 1]) #https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_2d.html
            self.weights[i] += self.deltas[i] * input_to_use.T * self.learning_rate

def main():
    X = pd.read_csv('../titanic/train_x.csv', sep=',', skiprows=1, names=['Pclass', 'Sex', 'Age', 'Parch', 'Fare'])
    y = pd.read_csv('../titanic/train_y.csv', sep=',', skiprows=1, names=['Survived'])

    X = one_hot_encoding(X, "Pclass")

    # # #regularization
    for column in X:
        X[column] = regularization(X[column])

    nn = NeuralNetwork(X, y, 3)
    nn.add_layer(X.shape[1],4) #X.shape[1] is num of input cols
    nn.add_layer(4,3)
    nn.add_layer(3,1)

    m = X.shape[0] #number of samples
    for i in range(100):#0000):
        sq_mean_error = 0
        cross_entropy = 0
        for j in range(m):
            nn.feedforward(j)
            sq_mean_error += nn.sq_mean_error
            cross_entropy += nn.cross_entropy
            nn.backprop(j)

    print("cost: square mean error: ",sq_mean_error[0])
    print("cost: cross-entropy: ",(cross_entropy/m))

    print("Weights: ",nn.weights)
if __name__ == "__main__":
    main()
