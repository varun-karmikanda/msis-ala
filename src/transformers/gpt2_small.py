import numpy as np
import numpy.typing as npt

class GPT2SmallParams:
    def __init__(self, d_e: int = 768, N_V: int = 50257, l_max: int=1024, d_attn=64, d_in=768, d_out=64, d_mid=64, head=12, d_out_h=768):
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
        self._W_o: npt.NDArray[np.float64] = np.random.rand(d_out_h, head * d_mid)
        self._b_o: npt.NDArray[np.float64] = np.random.rand(d_out_h)
    
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
    
    @property
    def W_o(self):
        return self._W_o
    
    @property
    def b_o(self):
        return self._b_o
    
def softmax(S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    S = S - S.max(axis=0, keepdims=True)
    S_exp = np.exp(S)
    softmax = S_exp / S_exp.sum(axis=0, keepdims=True)
    return softmax

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
) -> npt.NDArray[np.float64]:
    W_q, W_k, W_v = W_qkv
    b_q, b_k, b_v = b_qkv
    
    d_attn = W_q.shape[0]
    
    q = W_q @ e + b_q
    k = W_k @ e_t + b_k[:, None]
    v = W_v @ e_t + b_v[:, None]
    
    S = (k.T @ q) / np.sqrt(d_attn)
    
    alpha_t = softmax(S)
    
    v_tilde = v @ alpha_t
    return v_tilde

def attention(
    X: npt.NDArray[np.float64],
    Z: npt.NDArray[np.float64],
    W_qkv: npt.NDArray[np.float64],
    b_qkv: npt.NDArray[np.float64],
    mask: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    W_q, W_k, W_v = W_qkv
    b_q, b_k, b_v = b_qkv
    
    d_attn = W_q.shape[0]
    
    Q = W_q @ X + b_q[:, None]
    K = W_k @ Z + b_k[:, None]
    V = W_v @ Z + b_v[:, None]
    
    S = (K.T @ Q) / np.sqrt(d_attn)
    S = np.where(mask.astype(bool), S, -np.inf)
    
    V_tilde = V @ softmax(S)
    return V_tilde

def multi_head_attention(
    X: npt.NDArray[np.float64],
    Z: npt.NDArray[np.float64],
    W_qkv: npt.NDArray[np.float64],
    b_qkv: npt.NDArray[np.float64],
    mask: npt.NDArray[np.float64],
    heads: int,
    W_o: npt.NDArray[np.float64],
    b_o: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    Y_h = [attention(X, Z, W_qkv, b_qkv, mask) for _ in range(heads)]
    
    Y = np.concatenate(Y_h, axis=0)
    
    V_tilde = W_o @ Y + b_o[:, None]
    return V_tilde
    
        

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
    # print(single_query_attention(e, e_t, (W_q, W_k, W_v), (b_q, b_k, b_v)))
    
    X = np.random.rand(768, 10)
    Z = np.random.rand(768, 10)
    mask = np.triu(np.ones((10, 10)))
    # print(attention(X, Z, (W_q, W_k, W_v), (b_q, b_k, b_v), mask))
    
    heads = 12
    W_o = params.W_o
    b_o = params.b_o
    print(multi_head_attention(X, Z, (W_q, W_k, W_v), (b_q, b_k, b_v), mask, heads, W_o, b_o))