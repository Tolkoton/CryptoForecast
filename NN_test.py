from NN import *


class NN_test(NN):


    def __init__(self):
        self.NN_test_class = NN()
        #self.layer_dimensions = [2, 3, 1]
        self.layer_dimensions_real = [63, 5, 4, 3, 2, 1]

        #self.test_train_case_X = np.ones((2, 10))


    def test_init(self):
        print(self.NN_test_class)


    def test_sigmoid(self):
        pass


    def test_get_data(self):

        self.NN_test_class.get_data()

        print('X shape:', self.NN_test_class.X.shape)

        print( 'Y shape:', self.NN_test_class.Y.shape)

        print('train set X shape:', self.NN_test_class.train_set_X.shape)
        print('test set X shape:', self.NN_test_class.test_set_X.shape)

        print('train set Y shape:', self.NN_test_class.train_set_Y.shape)
        print('test set Y shape:', self.NN_test_class.test_set_Y.shape)


    def test_initialize_parameters_deep(self):
        '''
        Dimensions for test: size of x, 2 hidden layers, 1 output layer
        :return:
        '''

        parameters = self.NN_test_class.initialize_parameters_deep(self.layer_dimensions_real)

        for key in parameters.keys():
            print(key, parameters[key].shape)

        return parameters


    def test_linear_forward(self):
        '''
        Z is a matrix, were each Z(i) for each training example X(i) is stacked in columns.
        each X(i) is also in ith column
        :return:
        '''

        parameters = self.NN_test_class.initialize_parameters_deep(self.layer_dimensions_real)
        A = self.NN_test_class.train_set_X

        W = parameters['W1']
        b = parameters['b1']

        Z, cache_linear = self.NN_test_class.linear_forward(A, W, b)

        print('Z shape:', Z.shape)

        print('Cache A shape:', cache_linear[0].T.shape)
        #print('train set shape', self.NN_test_class.train_set_X.shape)

        print('Cache W shape:', cache_linear[1].shape)
        print('W shape:', W.shape)

        print('len cache_linear', len(cache_linear))


    def test_linear_activation_forward(self):

        parameters = self.NN_test_class.initialize_parameters_deep(self.layer_dimensions)
        A_prev = self.test_train_case_X

        W = parameters['W1']
        b = parameters['b1']

        A, cache_linear_activation = self.NN_test_class.linear_activation_forward(A_prev, W, b, 'linear')

        #print('A=', A, '\n linear_cache', cache_linear_activation[0], len(cache_linear_activation[0]), '\n activation_cache', cache_linear_activation[1])
        print('A shape:', A.shape, 'linear cache shapes: A shape:', cache_linear_activation[0][0].shape, 'W shape:', \
              cache_linear_activation[0][1].shape, 'b shape:', cache_linear_activation[0][2].shape, \
              'activation cache shape:', cache_linear_activation[1].shape)

        assert(cache_linear_activation[0][0].shape == cache_linear_activation[1].shape)

    def test_L_model_forward(self):
        parameters = self.NN_test_class.initialize_parameters_deep(self.layer_dimensions)
        AL, caches_list = self.NN_test_class.L_model_forward(self.test_train_case_X, parameters)

        print('AL shape', AL.shape)

        print('len caches_list', len(caches_list))
        print(caches_list[1])


    def test_cost(self):

        self.NN_test_class.get_data()
        parameters = self.NN_test_class.initialize_parameters_deep(self.layer_dimensions_real)
        AL = self.NN_test_class.L_model_forward(self.NN_test_class.train_set_X, parameters)[0]

        cost = self.NN_test_class.compute_cost(AL, self.NN_test_class.train_set_Y)

        print('cost =', cost)


    def test_linear_backward(self):

        self.NN_test_class.get_data()

        X = self.NN_test_class.train_set_X
        Y = self.NN_test_class.train_set_Y

        parameters = self.NN_test_class.initialize_parameters_deep(self.layer_dimensions_real)

        '''
        caches_list -- list of caches_list containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        '''
        AL, caches = self.NN_test_class.L_model_forward(X, parameters)
        print(len(caches))
        cache_Z = caches[1][0]



        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dA = dAL

        dZ = sigmoid_backward(dA, caches[1][1])

        grads = self.NN_test_class.linear_backward(dZ, cache_Z)

        print(grads)


    def test_linear_activation_backward(self):
        pass


    def test_model(self):
        self.NN_test_class.get_data()

        #X = self.NN_test_class.train_set_X
        #Y = self.NN_test_class.train_set_Y
        #Y = np.ones((self.NN_test_class.train_set_Y.shape))
        X = np.random.random((self.NN_test_class.train_set_X.shape))
        Y = np.random.randint(0, 2, self.NN_test_class.train_set_Y.shape)
        #print(Y)
        #parameters = self.NN_test_class.initialize_parameters_deep(self.layer_dimensions_real)

        self.NN_test_class.L_layer_model(X, Y, self.layer_dimensions_real, print_cost=True)


    #original tests

    def linear_backward_test_case(self):
        """
        z, linear_cache = (np.array([[-0.8019545 ,  3.85763489]]), (np.array([[-1.02387576,  1.12397796],
           [-1.62328545,  0.64667545],
           [-1.74314104, -0.59664964]]), np.array([[ 0.74505627,  1.97611078, -1.24412333]]), np.array([[1]]))
        """
        np.random.seed(1)
        dZ = np.random.randn(1, 2)
        A = np.random.randn(3, 2)
        W = np.random.randn(1, 3)
        b = np.random.randn(1, 1)
        linear_cache = (A, W, b)

        dA_prev, dW, db = self.NN_test_class.linear_backward(dZ, linear_cache)
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db))

        print('Expected output:')
        print('**dA_prev**	[[ 0.51822968 -0.19517421] [-0.40506361 0.15255393] [ 2.37496825 -0.89445391]]')
        print('**dW**	[[-0.10076895 1.40685096 1.64992505]]')
        print('**db**	[[ 0.50629448]]')

        #return dZ, linear_cache

x = NN_test()
# x.test_init()
#x.test_get_data()
x.test_initialize_parameters_deep()
#x.test_linear_forward()
#x.test_linear_activation_forward()
#x.test_L_model_forward()
#x.test_cost()
#x.test_linear_backward()
#x.test_model()
#x.linear_backward_test_case()