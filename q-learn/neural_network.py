import numpy
import math
import random
#from matplotlib import pyplot

class Neural:

    # constructor
    def __init__(self, n_input, n_hidden, n_output):
        self.hidden_weight = numpy.random.random_sample((n_hidden, n_input + 1))
        self.output_weight = numpy.random.random_sample((n_output, n_hidden + 1))
        self.hidden_momentum = numpy.zeros((n_hidden, n_input + 1))
        self.output_momentum = numpy.zeros((n_output, n_hidden + 1))
        self.output_num = n_output


# public method
    def train(self, X, T, epsilon, mu, epoch):
        self.error = numpy.zeros(epoch)
        N = X.shape[0]
        for epo in range(epoch):
            #print('nn.py/X[0,:]',X[0,:])
            for i in range(N):
                #print('nn.py/i',i)
                #print('nn.py/X[i,:]',X[i,:])
                x = X[i, :]
                t = T[i, :]

                self.__update_weight(x, t, epsilon, mu)

            self.error[epo] = self.__calc_error(X, T)


    def predict(self, X):
        N = X.shape[0]
        C = numpy.zeros(N).astype('int')
        #Y = numpy.zeros((N, T.shape[1]))
        Y = numpy.zeros((N, self.output_num))
        for i in range(N):
            x = X[i, :]
            z, y = self.__forward(x)

            Y[i] = y
            C[i] = y.argmax()

        return (C, Y)

# private method
    def __sigmoid(self, arr):
        return numpy.vectorize(lambda x: 1.0 / (1.0 + math.exp(-x)))(arr)


    def __forward(self, x):
        # z: output in hidden layer, y: output in output layer
        z = self.__sigmoid(self.hidden_weight.dot(numpy.r_[numpy.array([1]), x]))
        y = self.__sigmoid(self.output_weight.dot(numpy.r_[numpy.array([1]), z]))

        return (z, y)

    def __update_weight(self, x, t, epsilon, mu):
        z, y = self.__forward(x)

        # update output_weight
        output_delta = (y - t) * y * (1.0 - y)
        _output_weight = self.output_weight
        self.output_weight -= epsilon * output_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), z] - mu * self.output_momentum
        self.output_momentum = self.output_weight - _output_weight

        # update hidden_weight
        hidden_delta = (self.output_weight[:, 1:].T.dot(output_delta)) * z * (1.0 - z)
        _hidden_weight = self.hidden_weight
        self.hidden_weight -= epsilon * hidden_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), x]
        self.hidden_momentum = self.hidden_weight - _hidden_weight


    def __calc_error(self, X, T):
        N = X.shape[0]
        err = 0.0
        for i in range(N):
            x = X[i, :]
            t = T[i, :]

            z, y = self.__forward(x)
            err += (y - t).dot((y - t).reshape((-1, 1))) / 2.0

        return err


if __name__ == '__main__':

    #X = numpy.array([[0, 0.5,1], [0, 0.1,0.3], [1, 0,0.2], [1, 1,0.4]])
    #T = numpy.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    X = numpy.zeros((4,3))
    T = numpy.zeros((4,2))
    #Y = numpy.array([[1], [0], [0.2], [0.4]])
    N = X.shape[0] # number of data

    input_size = X.shape[1]
    hidden_size = X.shape[1]
    output_size = T.shape[1]

    epsilon = 0.1
    mu = 0.9
    epoch = 10000

    nn = Neural(input_size, hidden_size, output_size)
    nn.train(X, T, epsilon, mu, epoch)
    #nn.error_graph()

    #print('print',nn.predict(X,T))
    C, Y = nn.predict(X)
    print(C,Y)

    for i in range(N):
        x = X[i, :]
        y = Y[i, :]
        c = C[i]

        print x
        print y
        print c