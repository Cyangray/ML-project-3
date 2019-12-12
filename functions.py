import numpy as np
from math import floor

def make_onehot(a):
    '''takes a vector as input, and it returns it's onehot version as a numpy matrix'''
    uniques = np.unique(a)
    n_classes = len(uniques)
    a_onehot = np.zeros((len(a),n_classes))
    for i, elem in enumerate(a):
        for j, unique in enumerate(uniques):
            if elem == unique:
                a_onehot[i,j] = 1
                break
    return a_onehot

def inverse_onehot(a_onehot):
    '''The inverse onehot, taking a onehot vector and giving it's 1D version'''
    a = np.zeros(a_onehot.shape[0])
    for i in range(len(a)):
        a[i] = np.argmax(a_onehot[i,:])
    return a

def activation_function(x, activation):
    ''' A collection of activation functions for an argument x. 
    The activation argument can take the values 'sigmoid', 'softmax', 'linear',
    'tanh' or 'relu'.'''
    if activation == 'sigmoid':
        return sigmoid(x)
    elif activation == 'softmax':
        return softmax(x)
    elif activation == 'linear':
        return x
    elif activation == 'tanh':
        return np.tanh(x)
    elif activation == 'relu':
        return ReLU(x)
    else:
        print('unknown activation function')

def der_activation_function(x, activation):
    ''' derivatives of the activation functions in the above function. Same activation
    parameters.'''
    if activation == 'sigmoid':
        return sigmoid(x)*(1 - sigmoid(x))
    elif activation == 'softmax':
        return softmax(x)*(1 - softmax(x))
    elif activation == 'linear':
        return 1
    elif activation == 'tanh':
        return 1 - np.tanh(x)**2
    elif activation == 'relu':
        return np.heaviside(x, 0)

def ReLU(x):
    '''The ReLU activation function.'''
    return np.maximum(0,x)

def sigmoid(x):
    '''Sigmoid activation function used to map any real value between 0 and 1'''
    return 1. / (1. + np.exp(-x))

def softmax(x):
    '''The (normalized) softmax activation function.'''
    exp_term = np.exp(x)
    return exp_term / np.sum(exp_term, axis=1, keepdims=True)

def cost_function(t, a, method):
    '''function that returns the cost function. Argument t is the target, while
    a is the prediction.'''
    if method == 'cross_entropy':
        return -np.sum(t * np.log(a) + (1. - t) * np.log(1. - a))
    elif method == 'MSE':
        return 0.5 * np.sum((t - a)**2)
    
def dCda(t, a, method):
    '''function that returns the derivatives of the cost function. Argument t is 
    the target, while a is the prediction.'''
    if method == 'cross_entropy':
        return (a - t)/(a * (1 - a))
    elif method == 'MSE':
        return (a - t)
    

def franke_function(x,y):
    '''Generate values for the franke function'''
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def is_odd(num):
    '''Returns True if number is odd, False is even.'''
    return num & 0x1

def reduce4(A):
    '''Reduce the dimension of a matrix by four times, by only taking the first 
    value of every second for both axis'''
    
    A_rows = np.size(A,0)
    A_columns = np.size(A,1)
    A_rows_list = range(A_rows)
    A_columns_list = range(A_columns)
    
    B_rows = floor(A_rows/2)
    B_columns = floor(A_columns/2)
    
    B = np.zeros((B_rows, B_columns ))
    
    AtoB_rows = [A_rows_list[i]*2 for i in range(B_rows)]
    AtoB_columns = [A_columns_list[i]*2 for i in range(B_columns)]
    
    for i, row in enumerate(AtoB_rows):
        B[i,:] = A[row, AtoB_columns] 
    
    return B