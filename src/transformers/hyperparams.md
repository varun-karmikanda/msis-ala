# GPT-2 Small Architecture Parameters

| Parameter | Notation | Value | Description / Per-Head Dimensions |
| :--- | :--- | :--- | :--- |
| **Max Sequence Length** | $\ell_{\text{max}}$ | `1024` | Context window size (tokens) |
| **Transformer Layers** | $L$ | `12` | Total stacked blocks |
| **Attention Heads** | $H$ | `12` | Parallel heads per layer |
| **Vocabulary Size** | $N_V$ | `50257` | Token dictionary size |
| **Total Parameters** | $W_u$ | `124.4M` | Total un-embedding/model weights |
| **Model Embedding Width** | $d_e$ | `768` | Model hidden dimension ($d_{\text{model}}$) |
| **MLP Hidden Width** | $d_{\text{mlp}}$ | `3072` | Feed-forward inner dimension ($4 \times d_e$) |
| **Per-Head Attention Dim** | $d_{\text{attn}}$ ($d_{\text{mid}}$) | `64` | Subspace projection dimension ($d_e / H$) |
| **Input Dim (Per Head)** | $d_{\text{in}}$ | `768` | Input embedding size ($d_e$) |
| **Output Dim (Per Head)** | $d_{\text{out}}$ | `64` | Value projection size per head |

---

### Quick Copy: Raw Config Dictionary

```python
GPT2_SMALL_CONFIG = {
    "l_max": 1024,
    "L": 12,
    "H": 12,
    "N_V": 50257,
    "W_u": 124_400_000,
    "d_e": 768,
    "d_mlp": 3072,
    "d_attn": 64,   # d_mid per head (768 / 12)
    "d_in": 768,    # d_e per head
    "d_out": 64,    # per head
}
```