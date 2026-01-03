import sys
import os

# Allow importing from framework when running from addition_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from framework.autograd import Tensor
from addition_agent.dataset import AdditionDataset
from framework.model import TransformerModel
import argparse
import time
import addition_agent.config as config


class AdamOptimizer:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            grad = p.grad

            # m = beta1 * m + (1 - beta1) * g
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # v = beta2 * v + (1 - beta2) * g^2
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train():
    print("=== Training Tiny Autograd Transformer (with Adam) ===")

    # 1. Dataset
    MAX_DIGITS = 3
    dataset = AdditionDataset(max_digits=config.MAX_DIGITS)

    # Calculate sufficient max_len
    # Ensure enough room for input, output, and special tokens
    max_seq_len = dataset.max_input_len + dataset.max_output_len + 5

    # 2. Model
    model = TransformerModel(
        vocab_size=dataset.vocab_size,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        max_len=max_seq_len,
    )

    # Optimizer
    optimizer = AdamOptimizer(model.parameters(), lr=config.LEARNING_RATE)

    print(f"Model Parameters: {sum(p.data.size for p in model.parameters())}")
    print(f"Training for {config.MAX_STEPS} steps...")

    t0 = time.time()

    for i in range(config.MAX_STEPS):
        # --- Data Preparation ---
        src, tgt = dataset.get_batch(batch_size=config.BATCH_SIZE)
        full_input = np.concatenate([src, tgt[:, :-1]], axis=1)  # (B, L)

        targets_src = np.full_like(src, -100)
        targets_tgt = tgt[:, 1:]

        # Ignore loss for pad tokens
        targets_tgt = np.where(targets_tgt == dataset.pad_id, -100, targets_tgt)

        targets = np.concatenate([targets_src, targets_tgt], axis=1)  # (B, L)

        # --- Masks ---
        L = full_input.shape[1]

        # 1. Causal Mask
        causal_mask_val = np.triu(np.ones((L, L)), k=1) * -1e9

        # 2. Padding Mask
        is_pad = full_input == dataset.pad_id  # (B, L)
        pad_mask_val = is_pad[:, np.newaxis, np.newaxis, :] * -1e9

        # Combine
        combined_mask_val = causal_mask_val[np.newaxis, np.newaxis, :, :] + pad_mask_val
        mask = Tensor(combined_mask_val)

        # --- Forward Pass ---
        model.zero_grad()
        logits = model(full_input, mask)  # (B, L, V)

        # --- Loss Calculation ---
        loss = logits.reshape((-1, dataset.vocab_size)).cross_entropy(
            targets.reshape(-1)
        )

        # --- Backward Pass ---
        loss.backward()

        # --- Optimizer Step ---
        optimizer.step()

        # --- Logging ---
        if i % 10 == 0:
            print(f"Step {i} | Loss: {loss.data:.4f} | Time: {time.time()-t0:.2f}s")

        if i % 50 == 0:
            # Check last sample in batch
            pred_ids = np.argmax(logits.data[0, -3:, :], axis=-1)
            pred_tokens = [dataset.id_to_char.get(x, "") for x in pred_ids]
            print(f"  Example Prediction (Last 3 chars): {pred_tokens}")

    # Save Model
    model.save_weights(config.MODEL_FILENAME)


if __name__ == "__main__":
    train()
