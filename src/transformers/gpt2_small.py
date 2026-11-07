import numpy as np
import numpy.typing as npt

class GPT2SmallParams:
    
    def __init__(self, d_e: int = 768, N_V: int = 50257, l_max: int=1024):
        self._W_e: npt.NDArray[np.float64] = np.random.rand(d_e, N_V)
        self._W_p: npt.NDArray[np.float64] = np.random.rand(d_e, l_max)
    
    @property
    def W_e(self):
        return self._W_e
    
    @property
    def W_p(self):
        return self._W_p
    

def token_embedding(v: int, W_e: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return W_e[:, v]

def positional_embedding(l: int, W_p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return W_p[:, l]

if __name__ == "__main__":
    
    params = GPT2SmallParams(768, 50257, 1024)
    
    W_e = params.W_e
    v = 6767
    print(token_embedding(v, W_e))
    
    W_p = params._W_p
    l = 67
    print(positional_embedding(l, W_p))