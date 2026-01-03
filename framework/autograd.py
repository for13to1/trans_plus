import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.grad = np.zeros_like(self.data, dtype=np.float32)
        # Internal variables for autograd graph building
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Tensor(data={self.data}, op={self._op})"

    def backward(self):
        # Topological Sort
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    # --- Basic Operations ---

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            # Robust Broadcast handling
            grad_out = out.grad
            ndim_out = grad_out.ndim

            # Gradient for self
            ndim_self = self.data.ndim
            pad_self = (1,) * (ndim_out - ndim_self) + self.data.shape

            axes_self = []
            for i, (dim_out, dim_in) in enumerate(zip(grad_out.shape, pad_self)):
                if dim_in == 1 and dim_out > 1:
                    axes_self.append(i)

            # Sum over broadcasted dims
            grad_self = np.sum(grad_out, axis=tuple(axes_self))
            self.grad += grad_self.reshape(self.data.shape)

            # Gradient for other
            ndim_other = other.data.ndim
            pad_other = (1,) * (ndim_out - ndim_other) + other.data.shape

            axes_other = []
            for i, (dim_out, dim_in) in enumerate(zip(grad_out.shape, pad_other)):
                if dim_in == 1 and dim_out > 1:
                    axes_other.append(i)

            grad_other = np.sum(grad_out, axis=tuple(axes_other))
            other.grad += grad_other.reshape(other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            d_self = other.data * out.grad
            axis_self = tuple(range(d_self.ndim - self.data.ndim))
            if axis_self:
                self.grad += np.sum(d_self, axis=axis_self).reshape(self.data.shape)
            else:
                self.grad += d_self

            d_other = self.data * out.grad
            axis_other = tuple(range(d_other.ndim - other.data.ndim))
            if axis_other:
                other.grad += np.sum(d_other, axis=axis_other).reshape(other.data.shape)
            else:
                other.grad += d_other

        out._backward = _backward
        return out

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            # Handle Batched Matmul
            other_T = np.swapaxes(other.data, -1, -2)
            grad_self = out.grad @ other_T
            self.grad += grad_self

            self_T = np.swapaxes(self.data, -1, -2)
            grad_other = self_T @ out.grad

            if len(grad_other.shape) > len(other.data.shape):
                axes_to_sum = tuple(
                    range(len(grad_other.shape) - len(other.data.shape))
                )
                grad_other = np.sum(grad_other, axis=axes_to_sum)

            other.grad += grad_other

        out._backward = _backward
        return out

    def transpose(self, dim0, dim1):
        out = Tensor(np.swapaxes(self.data, dim0, dim1), (self,), "T")

        def _backward():
            self.grad += np.swapaxes(out.grad, dim0, dim1)

        out._backward = _backward
        return out

    def reshape(self, shape):
        old_shape = self.data.shape
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward():
            self.grad += out.grad.reshape(old_shape)

        out._backward = _backward
        return out

    # --- Activations ---

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        x_max = np.max(self.data, axis=axis, keepdims=True)
        exps = np.exp(self.data - x_max)
        sums = np.sum(exps, axis=axis, keepdims=True)
        probs = exps / sums

        out = Tensor(probs, (self,), "softmax")

        def _backward():
            y = out.data
            g = out.grad
            term1 = y * g
            term2 = y * np.sum(term1, axis=axis, keepdims=True)
            self.grad += term1 - term2

        out._backward = _backward
        return out

    def log_softmax(self, axis=-1):
        x_max = np.max(self.data, axis=axis, keepdims=True)
        shifted = self.data - x_max
        exps = np.exp(shifted)
        sums = np.sum(exps, axis=axis, keepdims=True)
        log_sums = np.log(sums)

        data = shifted - log_sums
        out = Tensor(data, (self,), "log_softmax")

        def _backward():
            softmax_val = np.exp(out.data)
            grad_sum = np.sum(out.grad, axis=axis, keepdims=True)
            self.grad += out.grad - softmax_val * grad_sum

        out._backward = _backward
        return out

    # --- Loss Helper ---
    def cross_entropy(self, targets):
        probs = self.log_softmax(axis=-1)
        batch_size, vocab_size = self.data.shape[0], self.data.shape[-1]
        probs_flat = probs.data.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        valid_mask = targets_flat >= 0
        valid_targets = targets_flat[valid_mask]
        valid_probs = probs_flat[valid_mask]

        if len(valid_targets) > 0:
            correct_logprobs = valid_probs[np.arange(len(valid_targets)), valid_targets]
            loss_val = -np.mean(correct_logprobs)
        else:
            loss_val = 0.0

        out = Tensor(loss_val, (probs,), "nll_loss")

        def _backward():
            N = len(valid_targets)
            if N == 0:
                pass
            else:
                grad_flat = np.zeros_like(probs_flat)
                grad_valid = np.zeros((N, vocab_size))
                grad_valid[np.arange(N), valid_targets] = -1.0 / N
                grad_flat[valid_mask] = grad_valid
                probs.grad += grad_flat.reshape(probs.data.shape)

        out._backward = _backward
        return out
