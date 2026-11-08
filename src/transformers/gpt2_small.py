import numpy as np
import numpy.typing as npt

class GPT2SmallParams:
    def __init__(self, d_e: int = 768, N_V: int = 50257, l_max: int=1024, d_attn=64, d_in=768, d_out=64):
        self._W_e: npt.NDArray[np.float64] = np.random.rand(d_e, N_V)
        self._W_p: npt.NDArray[np.float64] = np.random.rand(d_e, l_max)
        self._e: npt.NDArray[np.float64] = np.random.rand(d_in)
        self._e_t: npt.NDArray[np.float64] = np.random.rand(d_in)
        self._W_q: npt.NDArray[np.float64] = np.random.rand(d_attn, d_in)
        self._W_k: npt.NDArray[np.float64] = np.random.rand(d_attn, d_in)
        self._W_v: npt.NDArray[np.float64] = np.random.rand(d_out, d_in)
        self._b_q: npt.NDArray[np.float64] = np.random.rand(d_attn)
        self._b_k: npt.NDArray[np.float64] = np.random.rand(d_attn)
        self._b_v: npt.NDArray[np.float64] = np.random.rand(d_out)
    
    @property
    def W_e(self):
        return self._W_e
    
    @property
    def W_p(self):
        return self._W_p
    
    @property
    def W_q(self):
        return self._W_q
    
    @property
    def W_k(self):
        return self._W_k
    
    @property
    def W_v(self):
        return self._W_v
    
    @property
    def e(self):
        return self._e
    
    @property
    def e_t(self):
        return self._e_t
    
    @property
    def b_q(self):
        return self._b_q
    
    @property
    def b_k(self):
        return self._b_k
    
    @property
    def b_v(self):
        return self._b_v
    

def token_embedding(v: int, W_e: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    e = W_e[:, v]
    return e

def positional_embedding(l: int, W_p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    e_p =  W_p[:, l]
    return e_p

def single_query_attention(
    e: npt.NDArray[np.float64],
    e_t: npt.NDArray[np.float64],
    W_qkv: npt.NDArray[np.float64],
    b_qkv: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    W_q, W_k, W_v = W_qkv
    b_q, b_k, b_v = b_qkv
    
    d_attn = W_q.shape[0]
    
    q = W_q @ e + b_q
    k = W_k @ e_t + b_k[:, None]
    v = W_v @ e_t + b_v[:, None]
    
    logit = (q.T @ k) / np.sqrt(d_attn)
    scores = logit - np.max(logit)
    exp_scores = np.exp(scores)
    
    alpha_t = exp_scores / np.sum(exp_scores)
    
    return alpha_t @ v
    

if __name__ == "__main__":
    
    params = GPT2SmallParams(768, 50257, 1024, 64, 768)
    
    W_e = params.W_e
    v = 6767
    # print(token_embedding(v, W_e))
    
    W_p = params._W_p
    l = 67
    # print(positional_embedding(l, W_p))
    
    e = params.e
    e_t = params.e_t
    
    W_q = params.W_q
    W_k = params.W_k
    W_v = params.W_v
    
    b_q = params.b_q
    b_k = params.b_k
    b_v = params.b_v
    print(single_query_attention(e, e_t, (W_q, W_k, W_v), (b_q, b_k, b_v)))