from NN import *


class NN_test(NN):


    def __init__(self):
        self.NN_test_class = NN()

    @staticmethod
    def linear_forward_test_case():
        np.random.seed(1)
        """
        X = np.array([[-1.02387576, 1.12397796],
     [-1.62328545, 0.64667545],
     [-1.74314104, -0.59664964]])
        W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
        b = np.array([[1]])
        """
        A = np.random.randn(3, 2)
        W = np.random.randn(1, 3)
        b = np.random.randn(1, 1)

        return A, W, b

    def run_linear_forward_test_case(self):
        A, W, b = NN_test.linear_forward_test_case()
        Z, linear_cache = NN.linear_forward(A, W, b)
        print("Z = " + str(Z))
        print('Expected output:')
        print('**Z**	[[ 3.26295337 -1.23429987]]')

        return A, W, b


    def linear_activation_forward_test_case(self):
        """
        X = np.array([[-1.02387576, 1.12397796],
     [-1.62328545, 0.64667545],
     [-1.74314104, -0.59664964]])
        W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
        b = 5
        """
        np.random.seed(2)
        A_prev = np.random.randn(3, 2)
        W = np.random.randn(1, 3)
        b = np.random.randn(1, 1)
        return A_prev, W, b


    def run_linear_activation_forward_test_case(self):

        A_prev, W, b = self.linear_activation_forward_test_case()

        A, linear_activation_cache = NN.linear_activation_forward(A_prev, W, b, activation="sigmoid")
        print("With sigmoid: A = " + str(A))
        print('Expected output:')
        print('**With sigmoid: A **	[[ 0.96890023 0.11013289]]')



        A, linear_activation_cache = NN.linear_activation_forward(A_prev, W, b, activation="relu")
        print("With ReLU: A = " + str(A))
        print('Expected output:')
        print('**With ReLU: A **	[[ 3.43896131 0. ]]')


    def L_model_forward_test_case(self):
        """
        X = np.array([[-1.02387576, 1.12397796],
     [-1.62328545, 0.64667545],
     [-1.74314104, -0.59664964]])
        parameters = {'W1': np.array([[ 1.62434536, -0.61175641, -0.52817175],
            [-1.07296862,  0.86540763, -2.3015387 ]]),
     'W2': np.array([[ 1.74481176, -0.7612069 ]]),
     'b1': np.array([[ 0.],
            [ 0.]]),
     'b2': np.array([[ 0.]])}
        """
        np.random.seed(1)
        X = np.random.randn(4, 2)
        W1 = np.random.randn(3, 4)
        b1 = np.random.randn(3, 1)
        W2 = np.random.randn(1, 3)
        b2 = np.random.randn(1, 1)
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return X, parameters


    def run_L_model_forward_test_case(self):
        X, parameters = self.L_model_forward_test_case()
        AL, caches = NN.L_model_forward(X, parameters)
        print("AL = " + str(AL))
        print("Length of caches list = " + str(len(caches)))

        print('Expected output:')
        print('**AL**	[[ 0.17007265 0.2524272 ]]')
        print('**Length of caches list **	2')

    def compute_cost_test_case(self):
        Y = np.asarray([[1, 1, 1]])
        aL = np.array([[.8, .9, 0.4]])

        return Y, aL


    def run_compute_cost_test_case(self):
        Y, AL = self.compute_cost_test_case()

        print("cost = " + str(NN.compute_cost(AL, Y)))

        print('Expected output:')
        print('**cost**	0.41493159961539694')


    def L_model_backward_test_case(self):
        """
        X = np.random.rand(3,2)
        Y = np.array([[1, 1]])
        parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747]]), 'b1': np.array([[ 0.]])}
        aL, caches = (np.array([[ 0.60298372,  0.87182628]]), [((np.array([[ 0.20445225,  0.87811744],
               [ 0.02738759,  0.67046751],
               [ 0.4173048 ,  0.55868983]]),
        np.array([[ 1.78862847,  0.43650985,  0.09649747]]),
        np.array([[ 0.]])),
       np.array([[ 0.41791293,  1.91720367]]))])
       """
        np.random.seed(3)
        AL = np.random.randn(1, 2)
        Y = np.array([[1, 0]])

        A1 = np.random.randn(4, 2)
        W1 = np.random.randn(3, 4)
        b1 = np.random.randn(3, 1)
        Z1 = np.random.randn(3, 2)
        linear_cache_activation_1 = ((A1, W1, b1), Z1)

        A2 = np.random.randn(3, 2)
        W2 = np.random.randn(1, 3)
        b2 = np.random.randn(1, 1)
        Z2 = np.random.randn(1, 2)
        linear_cache_activation_2 = ((A2, W2, b2), Z2)

        caches = (linear_cache_activation_1, linear_cache_activation_2)

        return AL, Y, caches


    def run_L_model_backward_test_case(self):
        AL, Y_assess, caches = self.L_model_backward_test_case()
        grads = NN_test.L_model_backward(AL, Y_assess, caches)
        print("dW1 = " + str(grads["dW1"]))
        print("db1 = " + str(grads["db1"]))
        print("dA1 = " + str(grads["dA1"]))

        print("Expected output:")
        print('dW1	[[ 0.41010002 0.07807203 0.13798444 0.10502167] [ 0. 0. 0. 0. ] [ 0.05283652 0.01005865 0.01777766 0.0135308 ]]')
        print('db1	[[-0.22007063] [ 0. ] [-0.02835349]]')
        print('dA1	[[ 0. 0.52257901] [ 0. -0.3269206 ] [ 0. -0.32070404] [ 0. -0.74079187]]')


    def linear_activation_backward_test_case(self):
        """
        aL, linear_activation_cache = (np.array([[ 3.1980455 ,  7.85763489]]), ((np.array([[-1.02387576,  1.12397796], [-1.62328545,  0.64667545], [-1.74314104, -0.59664964]]), np.array([[ 0.74505627,  1.97611078, -1.24412333]]), 5), np.array([[ 3.1980455 ,  7.85763489]])))
        """
        np.random.seed(2)
        dA = np.random.randn(1, 2)
        A = np.random.randn(3, 2)
        W = np.random.randn(1, 3)
        b = np.random.randn(1, 1)
        Z = np.random.randn(1, 2)
        linear_cache = (A, W, b)
        activation_cache = Z
        linear_activation_cache = (linear_cache, activation_cache)

        return dA, linear_activation_cache


    def run_linear_activation_backward_test_case(self):

        AL, linear_activation_cache = NN_test.linear_activation_backward_test_case(self)

        dA_prev, dW, db = NN.linear_activation_backward(AL, linear_activation_cache, activation="sigmoid")
        print("sigmoid:")
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db) + "\n")

        print('Expected output with sigmoid:')
        print('dA_prev	[[ 0.11017994 0.01105339] [ 0.09466817 0.00949723] [-0.05743092 -0.00576154]]')
        print('dW	[[ 0.10266786 0.09778551 -0.01968084]]')
        print('db	[[-0.05729622]]')

        dA_prev, dW, db = NN.linear_activation_backward(AL, linear_activation_cache, activation="relu")
        print("\nRelu:")
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db))

        print('Expected output with relu:')
        print('dA_prev	[[ 0.44090989 0. ] [ 0.37883606 0. ] [-0.2298228 0. ]]')
        print('dW	[[ 0.44513824 0.37371418 -0.10478989]]')
        print('db	[[-0.20837892]]')

        def update_parameters_test_case():
            """
            parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747],
                [-1.8634927 , -0.2773882 , -0.35475898],
                [-0.08274148, -0.62700068, -0.04381817],
                [-0.47721803, -1.31386475,  0.88462238]]),
         'W2': np.array([[ 0.88131804,  1.70957306,  0.05003364, -0.40467741],
                [-0.54535995, -1.54647732,  0.98236743, -1.10106763],
                [-1.18504653, -0.2056499 ,  1.48614836,  0.23671627]]),
         'W3': np.array([[-1.02378514, -0.7129932 ,  0.62524497],
                [-0.16051336, -0.76883635, -0.23003072]]),
         'b1': np.array([[ 0.],
                [ 0.],
                [ 0.],
                [ 0.]]),
         'b2': np.array([[ 0.],
                [ 0.],
                [ 0.]]),
         'b3': np.array([[ 0.],
                [ 0.]])}
            grads = {'dW1': np.array([[ 0.63070583,  0.66482653,  0.18308507],
                [ 0.        ,  0.        ,  0.        ],
                [ 0.        ,  0.        ,  0.        ],
                [ 0.        ,  0.        ,  0.        ]]),
         'dW2': np.array([[ 1.62934255,  0.        ,  0.        ,  0.        ],
                [ 0.        ,  0.        ,  0.        ,  0.        ],
                [ 0.        ,  0.        ,  0.        ,  0.        ]]),
         'dW3': np.array([[-1.40260776,  0.        ,  0.        ]]),
         'da1': np.array([[ 0.70760786,  0.65063504],
                [ 0.17268975,  0.15878569],
                [ 0.03817582,  0.03510211]]),
         'da2': np.array([[ 0.39561478,  0.36376198],
                [ 0.7674101 ,  0.70562233],
                [ 0.0224596 ,  0.02065127],
                [-0.18165561, -0.16702967]]),
         'da3': np.array([[ 0.44888991,  0.41274769],
                [ 0.31261975,  0.28744927],
                [-0.27414557, -0.25207283]]),
         'db1': 0.75937676204411464,
         'db2': 0.86163759922811056,
         'db3': -0.84161956022334572}
            """
            np.random.seed(2)
            W1 = np.random.randn(3, 4)
            b1 = np.random.randn(3, 1)
            W2 = np.random.randn(1, 3)
            b2 = np.random.randn(1, 1)
            parameters = {"W1": W1,
                          "b1": b1,
                          "W2": W2,
                          "b2": b2}
            np.random.seed(3)
            dW1 = np.random.randn(3, 4)
            db1 = np.random.randn(3, 1)
            dW2 = np.random.randn(1, 3)
            db2 = np.random.randn(1, 1)
            grads = {"dW1": dW1,
                     "db1": db1,
                     "dW2": dW2,
                     "db2": db2}

            return parameters, grads

    def update_parameters_test_case(self):
        """
        parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747],
            [-1.8634927 , -0.2773882 , -0.35475898],
            [-0.08274148, -0.62700068, -0.04381817],
            [-0.47721803, -1.31386475,  0.88462238]]),
     'W2': np.array([[ 0.88131804,  1.70957306,  0.05003364, -0.40467741],
            [-0.54535995, -1.54647732,  0.98236743, -1.10106763],
            [-1.18504653, -0.2056499 ,  1.48614836,  0.23671627]]),
     'W3': np.array([[-1.02378514, -0.7129932 ,  0.62524497],
            [-0.16051336, -0.76883635, -0.23003072]]),
     'b1': np.array([[ 0.],
            [ 0.],
            [ 0.],
            [ 0.]]),
     'b2': np.array([[ 0.],
            [ 0.],
            [ 0.]]),
     'b3': np.array([[ 0.],
            [ 0.]])}
        grads = {'dW1': np.array([[ 0.63070583,  0.66482653,  0.18308507],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ]]),
     'dW2': np.array([[ 1.62934255,  0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ,  0.        ]]),
     'dW3': np.array([[-1.40260776,  0.        ,  0.        ]]),
     'da1': np.array([[ 0.70760786,  0.65063504],
            [ 0.17268975,  0.15878569],
            [ 0.03817582,  0.03510211]]),
     'da2': np.array([[ 0.39561478,  0.36376198],
            [ 0.7674101 ,  0.70562233],
            [ 0.0224596 ,  0.02065127],
            [-0.18165561, -0.16702967]]),
     'da3': np.array([[ 0.44888991,  0.41274769],
            [ 0.31261975,  0.28744927],
            [-0.27414557, -0.25207283]]),
     'db1': 0.75937676204411464,
     'db2': 0.86163759922811056,
     'db3': -0.84161956022334572}
        """
        np.random.seed(2)
        W1 = np.random.randn(3, 4)
        b1 = np.random.randn(3, 1)
        W2 = np.random.randn(1, 3)
        b2 = np.random.randn(1, 1)
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        np.random.seed(3)
        dW1 = np.random.randn(3, 4)
        db1 = np.random.randn(3, 1)
        dW2 = np.random.randn(1, 3)
        db2 = np.random.randn(1, 1)
        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return parameters, grads


    def run_update_parameters_test_case(self):
        parameters, grads = self.update_parameters_test_case()
        parameters = NN.update_parameters(parameters, grads, 0.1)

        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

        print('Expected Output:')
        print('W1	[[-0.59562069 -0.09991781 -2.14584584 1.82662008] [-1.76569676 -0.80627147 0.51115557 -1.18258802] \
         [-1.0535704 -0.86128581 0.68284052 2.20374577]]')
        print('b1	[[-0.04659241] [-1.28888275] [ 0.53405496]]')
        print('W2	[[-0.55569196 0.0354055 1.32964895]]')
        print('[[-0.84610769]]')







x = NN_test()
#x.run_linear_forward_test_case()
#x.run_linear_activation_forward_test_case()
#x.run_L_model_forward_test_case()
x.run_compute_cost_test_case()
#x.run_L_model_backward_test_case()
#x.run_linear_activation_backward_test_case()
#x.run_update_parameters_test_case()