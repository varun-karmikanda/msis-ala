import numpy as np

def attention(X, Z, W, mask):
    Wq = W[0]
    Wk = W[1]
    Wv = W[2]
    bq = W[3]
    bk = W[4]
    bv = W[5]

    # lx_1 = np.ones((1, X.shape[1]))
    lx_1 = np.ones(X.shape[1])

    K = Wk @ X + bk[:, None]
    Q = Wq @ Z + bq[:, None]
    V = Wv @ Z + bv[:, None]

    S = K.T @ Q / np.sqrt(datt)
    
    S = np.where(mask.astype(bool), S, -np.inf)
    
    S = S - S.max(axis=0, keepdims=True)
    
    P = np.exp(S)
    
    softmax_S = P / P.sum(axis=0, keepdims=True)

    return V @ softmax_S
    

def token_embedding(v, We):
    print(We[:, v])
    
def layer_norn(e, gamma, beta, epsilon=1e-5):
    m = np.mean(e)
    
    v = np.var(e)
    
    e_cap_intermidiate = (e - m) / np.sqrt(v + epsilon)
    
    e_cap = e_cap_intermidiate * gamma + beta
    
    return e_cap
    

if __name__ == "__main__":
    # Number of tokens(lx)
    lx = 2  

    # dimention of primary token sequence
    de = 768
    dx = 768
    dz = 768

    # dimention of the attention
    datt = 64

    # dimention of output
    dout = 64

    # Input vector
    X = np.random.rand(dx, lx)

    Wq = np.random.rand(datt, dx)
    bq = np.random.rand(datt)

    Wk = np.random.rand(datt, dz)
    bk = np.random.rand(datt)

    Wv = np.random.rand(dout, dz)
    bv = np.random.rand(dout)

    mask = np.triu(np.ones((lx, lx)))
    
    W = (Wq, Wk, Wv, bq, bk, bv)
    
    attention(X, X, W, mask)
    
    
    # Token embedding
    N_V = 50257
    v = 2
    
    We = np.random.rand(de, N_V)
    print(We)
    
    token_embedding(v, We)

    # layer norm
    e = np.random.rand(de)
    
    out = layer_norn(e, 1, 0)
    print(out.mean(), out.std())