import numpy as np
from autograd import Tensor
import pickle

class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def save_weights(self, path):
        # Flatten all parameters data into a list
        params_data = [p.data for p in self.parameters()]
        with open(path, 'wb') as f:
            pickle.dump(params_data, f)
        print(f"Weights saved to {path}")

    def load_weights(self, path):
        try:
            with open(path, 'rb') as f:
                params_data = pickle.load(f)

            my_params = self.parameters()
            if len(params_data) != len(my_params):
                print(f"Error: Parameter count mismatch. Saved: {len(params_data)}, Model: {len(my_params)}")
                return

            for p, saved_data in zip(my_params, params_data):
                p.data = saved_data
            print(f"Weights loaded from {path}")
        except FileNotFoundError:
            print("No weight file found, using random init.")


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        # Xavier/Kaiming Init
        limit = np.sqrt(6 / (in_features + out_features))
        self.weight = Tensor(np.random.uniform(-limit, limit, (in_features, out_features)), label='W')
        self.bias = Tensor(np.zeros(out_features), label='b') if bias else None

    def __call__(self, x):
        out = x.matmul(self.weight)
        if self.bias:
            out = out + self.bias
        return out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias else [])

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = Tensor(np.random.randn(num_embeddings, embedding_dim) * 0.1)

    def __call__(self, indices):
        # Indices is numpy array of ints
        # We need to gather rows from self.weight
        # Since Tensor currently doesn't support advanced slicing in graph,
        # we treat this as a Lookup.
        # dL/dW[idx] += dL/dOut

        # Forward is easy:
        out_data = self.weight.data[indices]
        out = Tensor(out_data, (self.weight,), 'embedding')

        # Standard closure for backward
        def _backward():
            # Add gradient to the specific rows
            # We iterate over unique indices to sum gradients?
            # Or just use numpy's add.at for efficiency

            # indices might be (batch, seq)
            # out.grad is (batch, seq, dim)

            # Flatten indices
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
        self.gamma = Tensor(np.ones(normalized_shape))
        self.beta = Tensor(np.zeros(normalized_shape))
        self.eps = eps

    def __call__(self, x):
        # x: [batch, len, dim]
        # mean/var over last dim
        # Since Tensor lib is simple, let's implement minimal manual backward for LN or use primitives
        # Using primitives:
        # mean = x.mean(...) -> Need to implement mean in Tensor
        # Let's simplify and assume x is (B, L, D) and we normalize over D

        # Manual Forward/Backward for LayerNorm is often cleaner in simple engines

        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        x_norm = (x.data - mean) / std

        out_data = self.gamma.data * x_norm + self.beta.data
        out = Tensor(out_data, (x, self.gamma, self.beta), 'layernorm')

        def _backward():
            # Primitive gradients for LN are standard but verbose.
            # d/dx ( gamma * (x-u)/s + beta )

            N = x.data.shape[-1]
            x_centered = x.data - mean

            dychat = out.grad * self.gamma.data

            # 1/sigma * ( I - 1/N * ones - x_norm*x_norm^T * 1/N ) ?
            # Standard efficient implementation:
            term1 = N * dychat
            term2 = np.sum(dychat, axis=-1, keepdims=True)
            term3 = x_norm * np.sum(dychat * x_norm, axis=-1, keepdims=True)

            dx = (1.0 / N) / std * (term1 - term2 - term3)

            self.beta.grad += np.sum(out.grad, axis=(0, 1)) # Sum over batch/seq
            self.gamma.grad += np.sum(out.grad * x_norm, axis=(0, 1))
            x.grad += dx

        out._backward = _backward
        return out

    def parameters(self):
        return [self.gamma, self.beta]

class MultiHeadAttention(Module):
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)

    def __call__(self, q, k, v, mask=None):
        # q, k, v are Tensors [Batch, Seq, Dim]
        batch_size = q.data.shape[0]

        # Linear Projections
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # Reshape for Heads
        # (B, S, H, Dk) -> (B, H, S, Dk)
        # Tensor doesn't support complex reshape/transpose chain efficiently in my minimal Autograd,
        # but I implemented reshape and transpose.
        # Let's try doing it carefully.

        # Function to split heads
        def split_heads(x):
            x = x.reshape((batch_size, -1, self.num_heads, self.d_k))
            x = x.transpose(1, 2) # (B, S, H, Dk) -> (B, H, S, Dk)
            return x

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # Scaled Dot Product
        # Q: (B, H, S, Dk), K: (B, H, S, Dk) -> K.T: (B, H, Dk, S) (Need swap last two)

        # Manual swapaxes for K
        # K transpose is harder because transpose only takes 2 dims in my generic impl?
        # My transpose: `np.swapaxes(self.data, dim0, dim1)`.
        K_t = K.transpose(-1, -2)

        scores = Q.matmul(K_t) # (B, H, S, S)

        # Scale?
        # Tensor doesn't have scalar div yet.
        # Impl scalar ops or hack it
        scale_factor = 1.0 / np.sqrt(self.d_k)
        # Hack: multiply by constant Tensor
        scores = scores * Tensor(np.array(scale_factor))

        # Masking
        if mask is not None:
            # Mask is typically (B, 1, 1, S) or (1, 1, S, S)
            # Add large negative to masked positions
            # Tensor __add__ handles broadcast
            # we need `scores + mask`
            scores = scores + mask

        attn = scores.softmax(axis=-1)

        context = attn.matmul(V) # (B, H, S, Dk)

        # Concat heads
        # (B, H, S, Dk) -> (B, S, H, Dk) -> (B, S, D)
        context = context.transpose(1, 2)
        context = context.reshape((batch_size, -1, self.d_model))

        return self.w_o(context)

    def parameters(self):
        return self.w_q.parameters() + self.w_k.parameters() + self.w_v.parameters() + self.w_o.parameters()

class TransformerBlock(Module):
    def __init__(self, d_model, num_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm((d_model,))
        self.norm2 = LayerNorm((d_model,))

        self.ff1 = Linear(d_model, d_ff)
        self.ff2 = Linear(d_ff, d_model)
        self.relu = ReLU()

    def __call__(self, x, mask=None):
        # Resid connection?
        # x + attn(x)
        # My Tensor needs __add__ check

        attn_out = self.attn(x, x, x, mask)
        x = self.norm1(x + attn_out)

        ff_out = self.ff2(self.relu(self.ff1(x)))
        x = self.norm2(x + ff_out)
        return x

    def parameters(self):
        return self.attn.parameters() + self.norm1.parameters() + self.norm2.parameters() + \
               self.ff1.parameters() + self.ff2.parameters()

class TransformerModel(Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers=2):
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_emb = Tensor(self.get_pe(20, d_model)) # Fixed length 20
        self.layers = [TransformerBlock(d_model, num_heads, d_model*4) for _ in range(num_layers)]
        self.fc_out = Linear(d_model, vocab_size)

    def get_pe(self, max_len, d_model):
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe[np.newaxis, :, :]

    def __call__(self, indices, mask=None):
        # indices: (B, S)
        x = self.embedding(indices) # (B, S, D)

        # Add PE (slice to current length)
        seq_len = x.data.shape[1]
        pe_slice = Tensor(self.pos_emb.data[:, :seq_len, :])
        x = x + pe_slice

        for layer in self.layers:
            x = layer(x, mask)

        return self.fc_out(x)

    def parameters(self):
        params = self.embedding.parameters() + self.fc_out.parameters()
        for l in self.layers:
            params += l.parameters()
        return params
