import numpy as np
np.set_printoptions(suppress=True)

def relu(x):
    x = np.asarray(x, dtype=float)
    
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    x = np.asarray(x, dtype=float)
    
    return np.maximum(alpha * x, x)

def gelu(x):
    x = np.asarray(x, dtype=float)
    
    tan_h = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))
    
    return 0.5 * x * (1 + tan_h)

def logits(X, W, b):
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    
    return X @ W + b

def layer_activation(X, W, b, activation_func):
    Z = logits(X, W, b)
    return activation_func(Z)

def softmax(z):
    z = np.asarray(z, dtype=float)
    
    e = np.exp(z - np.max(z))
    return e / np.sum(e)

def one_hot_encoding(x, num_classes):
    x = np.asarray(x, dtype=int)
    
    if np.any(x < 0) or np.any(x > num_classes):
        raise ValueError("Invalid class index")
    
    res = np.zeros((len(x), num_classes))
    res[np.arange(len(x)), x] = 1
    
    return res

def mean(x):
    # return np.mean(x)
    return np.sum(x) / len(x)

def standard_deviation(x):
    # return np.std(x)
    return np.sqrt(np.sum((x - np.mean(x)) ** 2) / len(x))

def layer_norm(e, gamma, beta, epsilon=1e-5):
    m = mean(e)
    v = np.var(e)
    
    e_hat = (e - m) / np.sqrt(v + epsilon)
    
    return e_hat * gamma + beta

def l2_normalization(x):
    return np.sqrt(np.dot(x, x))



array = [-3,-1,0,2,5]
array_np = np.array([-3,-1,0,2,5])

print(relu(array))
print(relu(array_np))

print(leaky_relu(array))
print(leaky_relu(array_np))

print(gelu(array))
print(gelu(array_np))
print(np.round(gelu(array), 4))

X1 = [
    [1, 2, 3],
    [2, 1, 2]
]

W1 = [
    [1, 2],
    [2, 1],
    [1, 1],
]

b1 = [1,2]

X = np.array([
    [1, 2, 3],
    [2, 1, 2]
])

W = np.array([
    [1, 2],
    [2, 1],
    [1, 1],
])

b = np.array([1,2])

print(logits(X, W, b))
print(logits(X1, W1, b1))

Z = logits(X, W, b)
print(softmax(Z))

print(softmax([1000, 1001, 1002]))

x = [0, 1, 4]
print(one_hot_encoding(x, 5))

print(mean(x))
print(standard_deviation(x))
    

print(layer_norm([3 ,5 ,2 ,8], 1.5, 0.5))

print(l2_normalization([3, 4]))
print(l2_normalization([3, -4, 2]))


print(layer_activation(
    [1, 2], 
    [
        [0.5, -1.0],
        [2.0,  0.5]
    ],
    [0.5, -3],
    relu
))

print(layer_activation(
    [1, 2], 
    [
        [0.5, -1.0],
        [2.0,  0.5]
    ],
    [0.5, -3],
    leaky_relu
))

print(layer_activation(
    [1, 2], 
    [
        [0.5, -1.0],
        [2.0,  0.5]
    ],
    [0.5, -3],
    gelu
))

