import numpy as np
import numpy.typing as npt

# Just to quickly differentiate(readability)
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]

class GPT2SmallParams:
    def __init__(self, d_e: int = 768, N_V: int = 50257, l_max: int=1024, d_attn=64, d_in=768, d_out=64, d_mid=64, H=12, d_out_h=768):
        self._W_e: Matrix = self.rand_init(d_e, N_V)
        self._W_p: Matrix = self.rand_init(d_e, l_max)
        self._e: Vector = self.rand_init(d_in)
        self._e_t: Vector = self.rand_init(d_in)
        self._W_o: Matrix = self.rand_init(d_out_h, H * d_mid)
        self._b_o: Vector = self.rand_init(d_out_h)
        self._W_qkv: tuple[Matrix, Matrix, Matrix] = (self.rand_init(d_attn, d_in), self.rand_init(d_attn, d_in), self.rand_init(d_out, d_in))
        self._b_qkv: tuple[Vector, Vector, Vector] = (self.rand_init(d_attn), self.rand_init(d_attn), self.rand_init(d_out))
        self._W: tuple[tuple[Matrix, Matrix, Matrix], ...] = tuple(
            (
                self.rand_init(d_attn, d_in),
                self.rand_init(d_attn, d_in),
                self.rand_init(d_out, d_in)
            )
            for _ in range(H)
        )
        self._b: tuple[tuple[Vector, Vector, Vector], ...] = tuple(
            (
                self.rand_init(d_attn),
                self.rand_init(d_attn),
                self.rand_init(d_out),
            )
            for _ in range(H)
        )
    
    @property
    def W_e(self):
        return self._W_e
    
    @property
    def W_p(self):
        return self._W_p
    
    @property
    def e(self):
        return self._e
    
    @property
    def e_t(self):
        return self._e_t
    
    @property
    def W_o(self):
        return self._W_o
    
    @property
    def b_o(self):
        return self._b_o
    
    @property
    def W_qkv(self):
        return self._W_qkv
    
    @property
    def b_qkv(self):
        return self._b_qkv
    
    @property
    def W(self):
        return self._W
    
    @property
    def b(self):
        return self._b
    
    @staticmethod
    def rand_init(rows: int, cols: int | None = None) -> Vector | Matrix:
        if cols is None:
            return np.random.rand(rows)
        return np.random.rand(rows, cols)
    
def softmax(S: Matrix) -> Matrix:
    S = S - S.max(axis=0, keepdims=True)
    S_exp = np.exp(S)
    softmax = S_exp / S_exp.sum(axis=0, keepdims=True)
    return softmax

def token_embedding(v: int, W_e: Matrix) -> Vector:
    e = W_e[:, v]
    return e

def positional_embedding(l: int, W_p: Matrix) -> Vector:
    e_p =  W_p[:, l]
    return e_p

def single_query_attention(
    e: Vector,
    e_t: Vector,
    W_qkv: Matrix,
    b_qkv: Matrix,
) -> Matrix:
    W_q, W_k, W_v = W_qkv
    b_q, b_k, b_v = b_qkv
    
    d_attn = W_q.shape[0]
    
    q = W_q @ e + b_q
    k = W_k @ e_t + b_k[:, None]
    v = W_v @ e_t + b_v[:, None]
    
    S = (q @ k) / np.sqrt(d_attn)
    
    alpha_t = softmax(S)
    
    v_tilde = v @ alpha_t
    return v_tilde

def attention(
    X: Matrix,
    Z: Matrix,
    W_qkv: Matrix,
    b_qkv: Vector,
    mask: Vector
) -> Matrix:
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
    X: Matrix,
    Z: Matrix,
    W: Matrix,
    b: Matrix,
    mask: Matrix,
    heads: int,
    W_o: Matrix,
    b_o: Vector,
) -> Matrix:
    H = len(W)
    
    Y_h = []
    
    
    for h in range(H):
        W_q, W_k, W_v = W[h]
        b_q, b_k, b_v = b[h]
        
        V_tilde = attention(X, Z, (W_q, W_k, W_v), (b_q, b_k, b_v), mask)
        
        Y_h.append(V_tilde)
    
    Y = np.concatenate(Y_h)
    
    V_tilde = W_o @ Y + b_o[:, None]
    return V_tilde        

if __name__ == "__main__":
    
    params = GPT2SmallParams(768, 50257, 1024, 64, 768)
    
    v = 6767
    # print(token_embedding(v, params.W_e))
    
    l = 67
    # print(positional_embedding(l, params.W_p))

    
    # print(single_query_attention(params.e, params.e_t, params.W_qkv, params._b_qkv))
    
    tokens = 67
    X = np.random.rand(768, tokens)
    Z = np.random.rand(768, tokens)
    mask = np.triu(np.ones((tokens, tokens)))
    # print(attention(X, Z, params.W_qkv, params.b_qkv, mask))
    
    H = 12
    print(multi_head_attention(X, Z, params.W, params.b, mask, H, params.W_o, params.b_o).shape)
    
    # W_qkv = params.W_qkv
    # print(len(W_qkv))
    # print(W_qkv[0].shape)
    
    
    
    # def x1(y):
    #     print(y)
    
    # def x2(*y):
    #     print(y)
        
    # x1((3,4))
    # x2(3,4)
    
    # W_qkv_mha = np.random.rand(H, 3, 64, 768)
    # print(W_qkv_mha.shape)
    
    # X = tuple(
    #     (
    #         x - 1,
    #         x,
    #         x + 1
    #     )
    #     for x in range(10)
    # )
    # print(X)
    
    # print((params.W[0][0]).shape)