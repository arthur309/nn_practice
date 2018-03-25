
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib as plt


# In[4]:


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters)
    
    for l in range(1, L): #???L ought to be half?
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], 
                parameters['b'+str(l)], activation = "relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], 
            parameters['b'+str(L)], activation = "sigmoid")
    caches.append(cache)
    assert(AL.shape == (1, X.shape[1]))
    
    return AL, caches

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -(np.divide(Y, AL))-(np.divide(1-Y, 1-AL)) #???
    current_cache = caches[L-1]      #caches最后一个元素标号是L-1
    grads["dA"+str(L)], grads["dW"+str(L)], grads["db"+str(L)] = linear_activation_backward(
                dAL, current_cache, "sigmoid")
        
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads["dA"+str(l+1)], grads["dW"+str(l+1)], grads["db"+str(l+1)] = linear_activation_backward(
                    grads["dA"+str(l+2)], current_cache, "relu")    
    return grads

def update_parameter(parameters, grads, learning_rate):
    L = len(parameters)
    
    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)]+grads["dW"+str(l+1)]*learning_rate
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)]+grads["db"+str(l+1)]*learning_rate
    
    return parameters
        


# In[ ]:


def aaa():
	a = 3
	print(a)
	return a

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters["b"+str(l)] = np.zeros(layer_dims[l], 1)
        assert(parameters["W"+str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters["b"+str(l)].shape == (layer_dims[l], 1))
    
    return parameters

def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev)+b
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    cache = (A_prev, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    if activation == "relu":
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)    
    
    return A,cache

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(np.dot(Y, AL.T)+np.dot((1-Y), (1-AL).T))/m
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


# In[3]:


def sigmoid(Z):
    A = 1.0/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):  
    A = np.maximum(0,Z)  
    assert (A.shape == Z.shape)  
    cache = Z  
    
    return A,cache  

def sigmoid_backward(dA, cache):
    Z = cache 
    A = 1.0/(1+np.exp(-Z))
    dZ = dA*A*(1-A)
    assert(dZ.shape == Z.shape)
    
    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z<0] = 0
    assert(dZ.shape == Z.shape)
    
    return dZ

