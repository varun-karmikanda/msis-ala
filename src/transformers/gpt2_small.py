import numpy as np
import numpy.typing as npt

def token_embedding(v: int, W_e: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return W_e[:, v]

def positional_embedding(l: int, W_p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return W_p[:, l]

if __name__ == "__main__":
    d_e = 768
    N_V = 50257
    l_max = 1024
    
    v = 6767
    W_e = np.random.rand(d_e, N_V)
    
    print(token_embedding(v, W_e))
    
    l = 67
    W_p = np.random.rand(d_e, l_max)
    
    print(positional_embedding(l, W_p))