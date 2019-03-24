import numpy as np
import pandas as pd
import math
from statistics import stdev, mean
import matplotlib.pyplot as plt

# activation function
# params:
#       x: the value to be evaluated
# returns: the evaluation of x
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

# backprop to adjust weights depending on the magnitude of the error
# params:
#       x: the value to be evaluated
# returns: the evaluation of x
def sigmoid_derivative(x):
    return x * (1.0 - x)

# Scales the given data using regularization
# params:
#   column: a set of values
# returns: The set of values scaled
def regularization(column):
    aux_column = []
    for value in column:
        aux_column.append((value-mean(column))/stdev(column))
    return aux_column

# Applies one_hot_encoding to the column_name from X
# params:
#   X: the DataFrame with the input data
#   column_name: the name of the column that wants to be 'one-hot-encoded'
# returns: The dataframe with the new columns of the one_hot_encoding
def one_hot_encoding(X, column_name):
    possible_values = X[column_name].unique()
    aux_matrix = [[0]*len(possible_values)]*len(X[column_name]) #matrix of zeros
    df = pd.DataFrame(aux_matrix, columns=possible_values)

    for i in range(len(X[column_name])):
        value = X[column_name].iloc[i]
        df.loc[i][value] = 1
    X = X.drop(column_name, axis=1)
    X = pd.concat([X, df], axis=1, join_axes=[X.index])
    return X

# Applies k-folds cross-validation
# params:
#   X: the full input DataFrame
#   y: the full input DataFrame
#   k: number of partitions
# returns: no return
def k_folds(X, y, k):
    accuracies = []
    chunks_x = split(X, int(len(X)/k))
    chunks_y = split(y, int(len(y)/k))
    for i in range(k):
        X = []
        y = []
        first_time = True
        for j in range(k):
            if(j!=i):
                if (first_time):
                    X = chunks_x[j]
                    y = chunks_y[j]
                    first_time = False
                else:
                    X=X.append(chunks_x[j], ignore_index=True)
                    y=y.append(chunks_y[j], ignore_index=True)
            else:
                cross_validation_x = chunks_x[j]
                cross_validation_y = chunks_y[j]
        f = open("k-folds/k-folds.txt","a")
        f.write('\nk = %d\n' %i)
        nn = NeuralNetwork(X, y, 3)
        epochs = 2
        nn.run_nn_simulation(X, y, epochs, f, True)
        accuracy = nn.check_accuracy(cross_validation_x, cross_validation_y)
        accuracies.append(accuracy)
        f.write("accuracy %f\n" %(accuracy))
    f.write('AVG accuracies %f' %mean(accuracies))

# Splits a dataframe creating new dataframes of size chunk_size
# params:
#   df: the dataframe
#   chunk_size: the size of the new dataframes
# return:
#   the new dataframes
# function retrieved from: http://yaoyao.codes/pandas/2018/01/23/pandas-split-a-dataframe-into-chunks
def split(df, chunk_size):
    indices = range(1 * chunk_size, (df.shape[0] // chunk_size + 1) * chunk_size, chunk_size)
    return np.split(df, indices)

# Cleans the dataframe given according to the results from the experimentation
# params:
#   df: the dataframe to be cleaned
# return:
#   df: the dataframe cleaned
def clean_dataset(df):
    #one hot encoding
    df = one_hot_encoding(df, "Pclass")
    #dimensionality reduction - consult atached files to see why I deleted this columns
    df = df.drop('Parch', axis=1)
    df = df.drop(3, axis=1)
    #regularization
    for column in df:
        df[column] = regularization(df[column])
    return df

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

    # Adds a layer to the nn
    # params:
    #   n_inputs: The input size (coming from the input layer or a previous hidden layer)
    #   n_neurons:The number of neurons in this layer.
    # return: no return
    def add_layer(self, n_inputs, n_neurons):
        self._layers.append(None)
        weights = np.random.rand(n_inputs, n_neurons)
        self.weights.append(weights)

    # Feedforwards the nn
    # params:
    #   id_row: the id corresponding to the current row from the input
    # return: no return
    def feedforward(self, id_row):
        current_x = self.input.iloc[id_row,:]
        for i in range(len(self._layers)):
            if (i == 0 ): #if first layer
                self._layers[i]= sigmoid(np.dot(current_x, self.weights[i]))
            else:
                self._layers[i] = sigmoid(np.dot(self._layers[i-1], self.weights[i]))
    # Prepares the data to feedforward the nn during the training
    # and calls the feedforward function
    # params:
    #   id_row: the id corresponding to the current row from the input
    #   n_values: the amount of rows in the dataframe
    # return: no return
    def _feedforward(self, id_row, n_values):
        current_y = self.y.iloc[id_row,:]['Survived']
        last_layer_index = len(self._layers)-1
        self.feedforward(id_row)
        self.sq_mean_error = ((current_y - self._layers[last_layer_index]) ** 2)/n_values
        self.cross_entropy  = -current_y*math.log(self._layers[last_layer_index])-(1-current_y)*math.log(1-self._layers[last_layer_index])

    # Backpropagation of the nn
    # params:
    #   id_row: the id corresponding to the current row from the input
    # return: no return
    def backprop(self, id_row):
        current_y = self.y.iloc[id_row,:]['Survived']
        current_x = self.input.iloc[id_row,:]
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

    # Evaluates the accuracy of the model
    # params:
    #   X: unseen input data to do cross-validation
    #   y: unseen results of the X
    # return:
    #   accuracy rate
    def check_accuracy(self, X, y):
        m = X.shape[0] #number of samples
        n_right = 0
        n_wrong = 0
        for j in range(m):
            result = self.test_value(j, X)
            if(y.iloc[j,:]['Survived'] == result):
                n_right += 1
            else:
                n_wrong += 1
        return n_right/(n_right+n_wrong)

    # Runs all the tests
    # params:
    #   test: the tests dataframe
    # return:
    #   no return
    def run_tests(self, test):
        m = test.shape[0] #number of samples
        for j in range(m):
            result = self.test_value(j, test)
            print(result)

    # Evaluates the given value
    # params:
    #   id_row: the id corresponding to the current row from the testing dataset
    #   x_test: the testing dataset
    # return: the result of the evaluation
    def test_value(self, id_row, x_test):
        # current_x = x_test.iloc[id_row,:]
        last_layer_index = len(self._layers)-1
        self.feedforward(id_row)
        estimation = self._layers[last_layer_index]
        result = 1 if estimation > 0.5 else 0
        return result #last layer


    # runs a simulation of the nn with the given input data X
    # params:
    #   X: the DataFrame with the input data
    #   y: the expected results
    #   n_epochs: number of epochs to be run
    #   f: file where all the info of the simulations will be saved
    def run_nn_simulation(self, X, y, n_epochs, f, print_weights = False):
        self.add_layer(X.shape[1],3) #X.shape[1] is num of input cols
        self.add_layer(3,2)
        self.add_layer(2,1)
        m = X.shape[0] #number of samples
        sq_mean_errors = []
        for i in range(n_epochs):
            sq_mean_error = 0
            cross_entropy = 0
            for j in range(m):
                self._feedforward(j, m)
                sq_mean_error += self.sq_mean_error
                cross_entropy += self.cross_entropy
                self.backprop(j)
            sq_mean_errors.append(sq_mean_error)
        print(sq_mean_errors)
        f.write("cost: square mean error: %f\n" %(sum(sq_mean_errors)/m))
        f.write("cost: cross-entropy: %f\n" %(cross_entropy/m))

        if (print_weights):
            f.write("weights: "+str(self.weights)+"\n")

        plt.clf()
        plt.plot(sq_mean_errors)
        plt.ylabel('Sq mean error')
        plt.xlabel('Epoch')
        plt.show()

def main():
    X = pd.read_csv('../titanic/train_x.csv', sep=',', skiprows=1, names=['Pclass', 'Sex', 'Age', 'Parch', 'Fare'])
    y = pd.read_csv('../titanic/train_y.csv', sep=',', skiprows=1, names=['Survived'])
    test = pd.read_csv('../titanic/test.csv', sep=',', skiprows=1, names=['Pclass', 'Sex', 'Age', 'Parch', 'Fare'])

    # clean_datasets
    X = clean_dataset(X)
    test = clean_dataset(test)

    #k-folds
    # k_folds(X, y, 10)
    m = X.shape[0] #number of samples

    f = open("training/training.txt","a")

    X_80_20 = split(X, int(0.8*m))
    y_80_20 = split(y, int(0.8*m))
    nn = NeuralNetwork(X, y, 3)
    epochs = 100
    nn.run_nn_simulation(X_80_20[0], y_80_20[0], epochs, f, True)
    nn.check_accuracy(X_80_20[1], y_80_20[1])

    nn.run_tests(test)

    print('\a')
if __name__ == "__main__":
    main()
