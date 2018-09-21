import pandas as pd
import matplotlib.pyplot as plt
from NN_activations import *

class NN(object):


    def __init__(self):
        self.iteration = 0


    def get_data(self):
        '''
        pd read dataset
        :return: None, creates self.train_set, self.test_set, self.m_train, self.m_test, self.X, self.Y
        '''

        trading_data = pd.read_csv('techind.csv', index_col= False)

        #save the dates
        open = trading_data['Open time']
        close = trading_data['Close time']
        self.open_close_time = pd.concat([open, close], axis = 1)

        #drop the dates from X to multiply matrices (its string). Drop 'Close its Y value'
        data_X = trading_data.drop('Open time', axis =1)
        data_X = data_X.drop('Close time', axis =1)
        self.X = data_X.drop('Close', axis=1).values

        # Model vectorization. X dimensions should be [n_x, m] =
        # = [number of features in a single training example, number of training examples].
        self.X = self.X.T

        self.Y = trading_data['Close'].values
        self.Y = self.Y.reshape(1, (len(self.Y)))

        m = len(trading_data)

        self.train_set_X = self.X[:, 0:(int(0.8*m))]
        self.test_set_X = self.X[:, (int(0.8*m)): m]

        self.train_set_Y = self.Y[:, 0:(int(0.8 * m))]
        self.test_set_Y = self.Y[:, (int(0.8 * m)): m]

        self.num_x = self.train_set_X.shape[0]


    @staticmethod
    def initialize_parameters_deep(layer_dimensions):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """


        parameters = {}
        L = len(layer_dimensions)  # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dimensions[l], layer_dimensions[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dimensions[l], 1))


            assert (parameters['W' + str(l)].shape == (layer_dimensions[l], layer_dimensions[l - 1]))
            assert (parameters['b' + str(l)].shape == (layer_dimensions[l], 1))

        return parameters


    @staticmethod
    def linear_forward(A, W, b):
        """
        Linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        cache_linear -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """
        #print('iteration')

        Z = np.dot(W, A) + b

        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache_linear = (A, W, b)

        return Z, cache_linear


    @staticmethod
    def linear_activation_forward(A_prev, W, b, activation):
        """
        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """

        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = NN.linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)

        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = NN.linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)

        # elif activation == 'linear':
        #     Z, linear_cache = NN.linear_forward(A_prev, W, b)
        #     A, activation_cache = linear(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache_linear_activation = (linear_cache, activation_cache)

        #print('linear_activation_forward \n','A_prev shape:', A_prev.shape, 'A.shape', A.shape, 'Z.shape:', Z.shape)
        return A, cache_linear_activation


    @staticmethod
    def L_model_forward(X, parameters):
        """
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches_list -- list of caches_list containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

        caches_list = []
        A = X
        L = len(parameters) // 2  # number of layers in the neural network


        for l in range(1, L):
            A_prev = A
            A, cache = NN.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation='relu')
            caches_list.append(cache)

        AL, cache = NN.linear_activation_forward(A,  parameters['W' + str(L)], parameters['b' + str(L)], activation='sigmoid')
        caches_list.append(cache)

        #print('AL shape:', AL.shape, 'X.shape', X.shape, '(1, X.shape[1]):', (1, X.shape[1]))

        assert (AL.shape == (1, X.shape[1]))


        return AL, caches_list


    @staticmethod
    def compute_cost(AL, Y):
        """
        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        #print('AL=', AL)

        m = Y.shape[1]

        logprobs = np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), (1 - Y))
        #print(logprobs)
        cost = -1 / m * np.sum(logprobs)

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

        assert (cost.shape == ())

        return cost


    @staticmethod
    def linear_backward(dZ, cache):
        """
        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, cache[0].T) / m
        #db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(cache[1].T, dZ)

        # print('type db', type(db))
        # db = db.astype(float)

        #print('b shape', b.shape, 'db shape', db.shape, db)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        #assert (isinstance(db, float))


        return dA_prev, dW, db


    @staticmethod
    def linear_activation_backward(dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        #print(len(cache))
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = NN.linear_backward(dZ, linear_cache)


        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = NN.linear_backward(dZ, linear_cache)


        return dA_prev, dW, db


    @staticmethod
    def L_model_backward(AL, Y, caches):
        """
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches)  # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        #print('AL=', AL)

        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        #dAL = np.nan_to_num(dAL)
        #print('type dAL', type(dAL), dAL)

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        current_cache = caches[-1]

        #print('type current cache1', type(current_cache[1]))


        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] =\
            NN.linear_activation_backward(dAL, current_cache, activation='sigmoid')

        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = NN.linear_activation_backward(grads['dA' + str(l + 2)], current_cache, activation='relu')
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp


        return grads


    @staticmethod
    def update_parameters(parameters, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
        """

        L = len(parameters) // 2  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]


        return parameters


    @staticmethod
    def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
        """
        Arguments:
        X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)
        costs = []  # keep track of cost

        # Parameters initialization.
        parameters = NN.initialize_parameters_deep(layers_dims)


        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = NN.L_model_forward(X, parameters)

            # Compute cost.
            cost = NN.compute_cost(AL, Y)

            # Backward propagation.
            grads = NN.L_model_backward(AL, Y, caches)

            # Update parameters.
            parameters = NN.update_parameters(parameters, grads, learning_rate)

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        return parameters