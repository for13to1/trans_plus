import numpy as np
from .autograd import Tensor
import pickle


class Module:
    def __init__(self):
        self.training = True

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def save_weights(self, path):
        params_data = [p.data for p in self.parameters()]
        with open(path, "wb") as f:
            pickle.dump(params_data, f)
        print(f"Weights saved to {path}")

    def load_weights(self, path):
        try:
            with open(path, "rb") as f:
                params_data = pickle.load(f)

            my_params = self.parameters()
            if len(params_data) != len(my_params):
                print(
                    f"Error: Parameter count mismatch. Saved: {len(params_data)}, Model: {len(my_params)}"
                )
                return

            for p, saved_data in zip(my_params, params_data):
                p.data = saved_data
            print(f"Weights loaded from {path}")
        except FileNotFoundError:
            print("No weight file found, using random init.")


class Dropout(Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def __call__(self, x):
        if not self.training or self.p == 0:
            return x

        # Create a mask of 1s and 0s
        # Scale by 1/(1-p) to maintain expected value
        scale = 1.0 / (1.0 - self.p)
        mask = np.random.binomial(1, 1.0 - self.p, size=x.data.shape) * scale

        # We treat mask as a constant Tensor (no gradient required for the mask itself)
        mask_tensor = Tensor(mask, label="dropout_mask")

        return x * mask_tensor


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, weight=None):
        super().__init__()
        if weight is not None:
            self.weight = weight
        else:
            # Xavier 初始化
            limit = np.sqrt(6 / (in_features + out_features))
            self.weight = Tensor(
                np.random.uniform(-limit, limit, (in_features, out_features)), label="W"
            )
        self.bias = Tensor(np.zeros(out_features), label="b") if bias else None

    def __call__(self, x):
        out = x.matmul(self.weight)
        if self.bias:
            out = out + self.bias
        return out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias else [])


class SharedLinear(Module):
    def __init__(self, d_model, vocab_size, shared_weight, bias=False):
        super().__init__()
        self._shared_weight = shared_weight
        self._vocab_size = vocab_size
        self._d_model = d_model
        self.bias = Tensor(np.zeros(vocab_size), label="b") if bias else None

    def __call__(self, x):
        shared_w = self._shared_weight.data
        transposed = np.swapaxes(shared_w, -1, -2)
        transposed_tensor = Tensor(
            transposed, _children=(self._shared_weight,), _op="shared_T"
        )
        out = x.matmul(transposed_tensor)
        if self.bias:
            out = out + self.bias
        return out

    def parameters(self):
        if self._shared_weight in [p for layer in [] for p in []]:
            pass
        seen = set()
        params = [self._shared_weight]
        seen.add(id(self._shared_weight.data))
        if self.bias:
            params.append(self.bias)
        return params


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, weight=None):
        super().__init__()
        if weight is not None:
            self.weight = weight
        else:
            self.weight = Tensor(np.random.randn(num_embeddings, embedding_dim) * 0.1)

    def __call__(self, indices):
        out_data = self.weight.data[indices]
        out = Tensor(out_data, (self.weight,), "embedding")

        def _backward():
            flat_indices = indices.reshape(-1)
            flat_grad = out.grad.reshape(-1, out.grad.shape[-1])
            np.add.at(self.weight.grad, flat_indices, flat_grad)

        out._backward = _backward
        return out

    def parameters(self):
        return [self.weight]


class ReLU(Module):
    def __call__(self, x):
        return x.relu()


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.gamma = Tensor(np.ones(normalized_shape))
        self.beta = Tensor(np.zeros(normalized_shape))
        self.eps = eps

    def __call__(self, x):
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        x_norm = (x.data - mean) / std

        out_data = self.gamma.data * x_norm + self.beta.data
        out = Tensor(out_data, (x, self.gamma, self.beta), "layernorm")

        def _backward():
            N = x.data.shape[-1]
            x_centered = x.data - mean
            dychat = out.grad * self.gamma.data
            term1 = N * dychat
            term2 = np.sum(dychat, axis=-1, keepdims=True)
            term3 = x_norm * np.sum(dychat * x_norm, axis=-1, keepdims=True)
            dx = (1.0 / N) / std * (term1 - term2 - term3)

            self.beta.grad += np.sum(out.grad, axis=(0, 1))
            self.gamma.grad += np.sum(out.grad * x_norm, axis=(0, 1))
            x.grad += dx

        out._backward = _backward
        return out

    def parameters(self):
        return [self.gamma, self.beta]


class MultiHeadAttention(Module):
    """
    Multi-Head Attention mechanism as described in the Transformer paper.

    The attention mechanism maps a query and a set of key-value pairs to an output.
    Multi-Head Attention performs this function in parallel for `num_heads` times.

    Args:
        d_model: The number of expected features in the input (vector size).
        num_heads: The number of heads in the multiheadattention models.
        dropout: The dropout probability.

    Attributes:
        d_k (int): Dimensionality of the key/query vectors ($d_{model} / h$).
        w_q (Linear): Linear projection for Query ($W^Q$).
        w_k (Linear): Linear projection for Key ($W^K$).
        w_v (Linear): Linear projection for Value ($W^V$).
        w_o (Linear): Linear projection for Output ($W^O$).
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / np.sqrt(self.d_k)

        self.w_q = Linear(d_model, d_model, bias=True)
        self.w_k = Linear(d_model, d_model, bias=True)
        self.w_v = Linear(d_model, d_model, bias=True)
        self.w_o = Linear(d_model, d_model, bias=True)
        self.dropout = Dropout(dropout)

    def __call__(self, q, k, v, mask=None):
        """
        Forward pass for Multi-Head Attention.

        Args:
            q: Queries tensor of shape (Batch, Seq_q, d_model)
            k: Keys tensor of shape (Batch, Seq_k, d_model)
            v: Values tensor of shape (Batch, Seq_v, d_model)
            mask: Optional mask tensor (e.g., (Batch, 1, 1, Seq_k)) for causal/padding masking.

        Returns:
            Tensor of shape (Batch, Seq_q, d_model) representing the attended output.
        """
        batch_size = q.data.shape[0]

        # Linear Projections
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # Split Heads
        def split_heads(x):
            # x:
            # [batch_size, seq_len, d_model] ->
            # [batch_size, seq_len, num_heads, d_model // num_heads]
            # [batch_size, num_heads, seq_len, d_model // num_heads]
            x = x.reshape((batch_size, -1, self.num_heads, self.d_k))
            x = x.transpose(1, 2)  # (B, H, S, Dk)
            return x

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # Scaled Dot Product
        K_t = K.transpose(-1, -2)
        scores = Q.matmul(K_t) * Tensor(np.array(self.scale))

        if mask is not None:
            scores = scores + mask

        attn = scores.softmax(axis=-1)
        attn = self.dropout(attn)

        context = attn.matmul(V)

        # Concat heads
        context = context.transpose(1, 2)
        context = context.reshape((batch_size, -1, self.d_model))
        # [batch_size, num_heads, seq_len, d_model // num_heads] ->
        # [batch_size, seq_len, num_heads, d_model // num_heads] ->
        # [batch_size, seq_len, d_model]

        return self.w_o(context)

    def parameters(self):
        return (
            self.w_q.parameters()
            + self.w_k.parameters()
            + self.w_v.parameters()
            + self.w_o.parameters()
        )

    def train(self):
        super().train()
        self.dropout.train()

    def eval(self):
        super().eval()
        self.dropout.eval()


class PositionwiseFeedForward(Module):
    """
    Position-wise Feed-Forward Networks.

    A two-layer fully connected neural network applied to each position separately and identically.
    Equation: FFN(x) = max(0, xW1 + b1)W2 + b2

    Args:
        d_model: Dimensionality of model input/output.
        d_ff: Inner-layer dimensionality ($d_{ff}$, typically 4 * d_model).
        dropout: Random dropout probability.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = Linear(d_model, d_ff, bias=True)
        self.w_2 = Linear(d_ff, d_model, bias=True)
        self.dropout = Dropout(dropout)
        self.relu = ReLU()

    def __call__(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))

    def parameters(self):
        return self.w_1.parameters() + self.w_2.parameters()

    def train(self):
        super().train()
        self.dropout.train()

    def eval(self):
        super().eval()
        self.dropout.eval()


class EncoderLayer(Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm((d_model,))
        self.norm2 = LayerNorm((d_model,))
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def __call__(self, x, mask=None):
        # Sublayer 1: Self-Attention
        attn_out = self.self_attn(x, x, x, mask)
        # Self-Attention 的核心特征：Q=K=V=x。
        # 模型在理解当前词（x）时，Query 是它自己，去同一个序列（x）中查找信息（Key），并聚合对应的值（Value）。
        x = self.norm1(x + self.dropout1(attn_out))

        # Sublayer 2: FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

    def parameters(self):
        return (
            self.self_attn.parameters()
            + self.ffn.parameters()
            + self.norm1.parameters()
            + self.norm2.parameters()
        )

    def train(self):
        super().train()
        self.self_attn.train()
        self.ffn.train()
        self.dropout1.train()
        self.dropout2.train()

    def eval(self):
        super().eval()
        self.self_attn.eval()
        self.ffn.eval()
        self.dropout1.eval()
        self.dropout2.eval()


class DecoderLayer(Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = LayerNorm((d_model,))
        self.norm2 = LayerNorm((d_model,))
        self.norm3 = LayerNorm((d_model,))

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def __call__(self, x, memory, src_mask=None, tgt_mask=None):
        # Sublayer 1: Masked Self-Attention
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Sublayer 2: Cross-Attention
        # Query comes from decoder (x), Key and Value from encoder (memory)
        attn_out = self.cross_attn(x, memory, memory, src_mask)
        # Cross-Attention 的核心特征：Q=x, K=V=enc_output。
        # 模型在理解当前词（x）时，Query 是它自己，去编码器的输出（enc_output）中查找信息（Key），并聚合对应的值（Value）。
        x = self.norm2(x + self.dropout2(attn_out))

        # Sublayer 3: FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x

    def parameters(self):
        return (
            self.self_attn.parameters()
            + self.cross_attn.parameters()
            + self.ffn.parameters()
            + self.norm1.parameters()
            + self.norm2.parameters()
            + self.norm3.parameters()
        )

    def train(self):
        super().train()
        self.self_attn.train()
        self.cross_attn.train()
        self.ffn.train()
        self.dropout1.train()
        self.dropout2.train()
        self.dropout3.train()

    def eval(self):
        super().eval()
        self.self_attn.eval()
        self.cross_attn.eval()
        self.ffn.eval()
        self.dropout1.eval()
        self.dropout2.eval()
        self.dropout3.eval()


class Encoder(Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ]

    def __call__(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

    def parameters(self):
        params = []
        for l in self.layers:
            params += l.parameters()
        return params

    def train(self):
        super().train()
        for l in self.layers:
            l.train()

    def eval(self):
        super().eval()
        for l in self.layers:
            l.eval()


class Decoder(Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ]

    def __call__(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

    def parameters(self):
        params = []
        for l in self.layers:
            params += l.parameters()
        return params

    def train(self):
        super().train()
        for l in self.layers:
            l.train()

    def eval(self):
        super().eval()
        for l in self.layers:
            l.eval()


class Transformer(Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_len=5000,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model

        self.shared_weight = Tensor(
            np.random.randn(tgt_vocab_size, d_model) * 0.1, label="shared_embedding"
        )

        self.src_embedding = Embedding(
            src_vocab_size, d_model, weight=self.shared_weight
        )
        self.tgt_embedding = Embedding(
            tgt_vocab_size, d_model, weight=self.shared_weight
        )

        self.pos_emb = Tensor(self.get_pe(max_len, d_model))

        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)

        self.fc_out = SharedLinear(
            d_model, tgt_vocab_size, self.shared_weight, bias=False
        )

        self.dropout = Dropout(dropout)

    def get_pe(self, max_len, d_model):
        """
        Transformer 的位置编码，就是把自然语言中的“词序”，编码成了一组在复平面上以不同速度旋转的“时钟”。
        通过比较时钟的相位差（旋转角度），模型就能精确地算出词与词之间的相对距离，这与信号处理中利用相位来解调信息的原理如出一辙。

        :param max_len: The maximum sequence length (context window) the model can handle.
        :param d_model: The hidden dimension of the model (size of the embedding vector for each token).
        """
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]  # shape=(max_len, 1)

        # 把 $a^b$ 转化为 $\exp(b \cdot \ln a)$:
        # div_term = 1 / (10000 ^ (2i / d_model))
        #          = exp( -log(10000) * (2i / d_model) )
        # Use simple exp/log for better numerical stability compared to direct power division
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 1::2] = np.cos(position * div_term)

        # Add batch dimension [1, max_len, d_model] for broadcasting
        return pe[np.newaxis, :, :]

    def encode(self, src, src_mask=None):
        # src: (B, S)
        x = self.src_embedding(src) * Tensor(np.array(np.sqrt(self.d_model)))
        seq_len = x.data.shape[1]
        x = x + Tensor(self.pos_emb.data[:, :seq_len, :])
        x = self.dropout(x)
        return self.encoder(x, src_mask)

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        # tgt: (B, S)
        x = self.tgt_embedding(tgt) * Tensor(np.array(np.sqrt(self.d_model)))
        seq_len = x.data.shape[1]
        x = x + Tensor(self.pos_emb.data[:, :seq_len, :])
        x = self.dropout(x)
        return self.decoder(x, memory, src_mask, tgt_mask)

    def __call__(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.fc_out(output)

    def parameters(self):
        all_params = self.encoder.parameters() + self.decoder.parameters()
        seen_data_ids = set()
        unique_params = []
        for p in all_params:
            data_id = id(p.data)
            if data_id not in seen_data_ids:
                seen_data_ids.add(data_id)
                unique_params.append(p)
        if id(self.shared_weight.data) not in seen_data_ids:
            unique_params = [self.shared_weight] + unique_params
        return unique_params

    def train(self):
        super().train()
        self.encoder.train()
        self.decoder.train()
        self.dropout.train()

    def eval(self):
        super().eval()
        self.encoder.eval()
        self.decoder.eval()
        self.dropout.eval()
