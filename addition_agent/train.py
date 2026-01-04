#!/usr/bin/env python3
import sys
import os

# Allow importing from framework when running from addition_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from framework.autograd import Tensor
from addition_agent.dataset import AdditionDataset
from framework.model import Transformer
from addition_agent import config
import time


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


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.parameters:
            pass
        self.optimizer.lr = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        # lrate = d_model^-0.5 * min(step_num^-0.5, step_num * warmup_steps^-1.5)
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )


def train():
    print("=== Training autograd transformer using config.py ===")

    # 1. Dataset
    dataset = AdditionDataset(max_digits=config.MAX_DIGITS)

    # 2. Model with configuration from config.py
    model = Transformer(
        src_vocab_size=dataset.vocab_size,
        tgt_vocab_size=dataset.vocab_size,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        d_ff=config.D_FF,
        max_len=dataset.max_input_len + dataset.max_output_len + 10,
        dropout=0.1,
    )

    # Analyze actual model usage
    from framework.analyze import analyze_parameters, analyze_memory_usage

    analyze_parameters(model)
    analyze_memory_usage(
        d_model=config.D_MODEL,
        d_ff=config.D_FF,
        num_layers=config.NUM_LAYERS,
        vocab_size=dataset.vocab_size,
    )

    # Optimizer & Scheduler
    base_optimizer = AdamOptimizer(
        model.parameters(), lr=0.0, beta1=0.9, beta2=0.999, eps=1e-9
    )
    optimizer = NoamOpt(config.D_MODEL, 1, config.WARMUP_STEPS, base_optimizer)

    print(f"Model Parameters: {sum(p.data.size for p in model.parameters())}")
    print(f"Training for {config.MAX_STEPS} steps...")

    t0 = time.time()

    try:
        for i in range(config.MAX_STEPS):
            # --- Data Preparation ---
            src_batch, tgt_batch = dataset.get_batch(batch_size=config.BATCH_SIZE)

            decoder_input = tgt_batch[:, :-1]
            target_output = tgt_batch[:, 1:]

            # --- Masks ---
            src_is_pad = src_batch == dataset.pad_id
            src_mask = src_is_pad[:, np.newaxis, np.newaxis, :] * -1e9

            B, T = decoder_input.shape
            causal_mask = np.triu(np.ones((T, T)), k=1) * -1e9
            tgt_is_pad = decoder_input == dataset.pad_id
            tgt_pad_mask = tgt_is_pad[:, np.newaxis, np.newaxis, :] * -1e9
            tgt_mask = Tensor(causal_mask[np.newaxis, np.newaxis, :, :] + tgt_pad_mask)
            src_mask = Tensor(src_mask)

            # --- Forward Pass ---
            model.zero_grad()
            model.train()

            logits = model(src_batch, decoder_input, src_mask, tgt_mask)  # (B, T, V)

            # --- Simple Cross Entropy Loss ---
            flat_logits = logits.reshape((-1, dataset.vocab_size))
            flat_targets = target_output.reshape(-1)

            loss = flat_logits.cross_entropy(flat_targets)

            # --- Backward Pass ---
            loss.backward()

            # --- Optimizer Step ---
            optimizer.step()

            # --- Logging ---
            if i % 10 == 0:
                lr = optimizer.optimizer.lr
                print(
                    f"Step {i} | Loss: {loss.data:.4f} | LR: {lr:.6f} | Time: {time.time()-t0:.2f}s"
                )

            if i % 100 == 0:
                # Check last sample in batch
                model.eval()
                test_src = src_batch[0:1]  # (1, L)
                pred_tokens = []

                memory = model.encode(test_src, src_mask=None)
                curr_tgt = np.array([[dataset.sos_id]])

                for _ in range(dataset.max_output_len):
                    curr_len = curr_tgt.shape[1]
                    causal_mask_inf = np.triu(np.ones((curr_len, curr_len)), k=1) * -1e9
                    curr_mask = Tensor(causal_mask_inf[np.newaxis, np.newaxis, :, :])

                    out = model.decode(
                        curr_tgt, memory, src_mask=None, tgt_mask=curr_mask
                    )
                    next_token_logits = model.fc_out(out).data[:, -1, :]  # (1, V)
                    next_token = np.argmax(next_token_logits, axis=-1)

                    pred_tokens.append(dataset.id_to_char.get(next_token[0], ""))
                    curr_tgt = np.concatenate(
                        [curr_tgt, next_token[:, np.newaxis]], axis=1
                    )

                    if next_token[0] == dataset.eos_id:
                        break

                src_str = dataset.decode(test_src[0])
                print(f"  Example: {src_str} = {''.join(pred_tokens)}")
                model.train()

        print("Training completed successfully!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current weights...")
    finally:
        model.save_weights(config.MODEL_FILENAME)


if __name__ == "__main__":
    train()
