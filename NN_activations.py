import numpy as np
from scipy.special import expit



def sigmoid(Z):
    """
    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    #A = np.nan_to_num(A)
    #print(A)

    #A = 1 / (1 + expit(-Z))
    #A = np.nan_to_num(A)
    cache_Z = Z

    return A, cache_Z



def sigmoid_backward(dA, cache_Z):
    """
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache_Z


    s = 1 / (1 + np.exp(-Z))

    #s = 1 / (1 + expit(-Z))

    dZ = dA * s * (1 - s)
    #print(dZ.shape, 'da shape', dA.shape)

    assert (dZ.shape == Z.shape)

    return dZ



def relu(Z):
    """
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache_Z = Z

    return A, cache_Z



def relu_backward(dA, cache_Z):
    """
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache_Z
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    #print('dZ=', dZ)

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ



# def linear(Z):
#     """
#     Arguments:
#     Z -- Output of the linear layer, of any shape
#     Returns:
#     A -- Post-activation parameter, of the same shape as Z
#     cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
#     """
#     #print('linear(Z)', 'Z shape', Z.shape)
#     A = Z
#
#     assert (A.shape == Z.shape)
#
#     cache_Z = Z
#
#     return A, cache_Z



# def linear_backward(dA, cache_Z):
#     """
#     Arguments:
#     dA -- post-activation gradient, of any shape
#     cache -- 'Z' where we store for computing backward propagation efficiently
#     Returns:
#     dZ -- Gradient of the cost with respect to Z
#     """
#
#     Z = cache_Z
#     dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
#
#     assert (dZ.shape == Z.shape)
#
#     return dZ