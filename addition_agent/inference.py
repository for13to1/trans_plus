import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from framework.autograd import Tensor
from addition_agent.dataset import AdditionDataset
from framework.model import Transformer
import argparse
import addition_agent.config as config


def greedy_decode(model, dataset, problem_str, max_len=10):
    # Encode Input
    # "12+34" -> [1, 2, +, 3, 4]
    src_ids = dataset.encode(problem_str)

    # Needs to be a batch of 1
    src_batch = np.array([src_ids])  # (1, L)
    # Ideally should pad if we used batching, but for 1 it's fine.

    # 1. Encode
    # Create mask if needed (no padding here so None is fine or full ones)
    model.eval()
    memory = model.encode(src_batch, src_mask=None)

    # 2. Decode Loop
    # Start with <sos>
    current_tgt = [dataset.sos_id]
    result = []

    for _ in range(max_len):
        # Prepare input tensor
        tgt_batch = np.array([current_tgt])  # (1, T)

        # Causal Mask
        T = tgt_batch.shape[1]
        causal_mask_val = np.triu(np.ones((T, T)), k=1) * -1e9
        tgt_mask = Tensor(causal_mask_val[np.newaxis, np.newaxis, :, :])

        # Forward Decoder
        out = model.decode(tgt_batch, memory, src_mask=None, tgt_mask=tgt_mask)

        # Project to Vocab
        logits = model.fc_out(out)  # (1, T, V)

        # Get last token prediction
        last_logits = logits.data[0, -1, :]
        next_id = np.argmax(last_logits)

        if next_id == dataset.eos_id:
            break

        result.append(next_id)
        current_tgt.append(next_id)

        # Security break
        if len(result) > dataset.max_output_len + 5:
            break

    return dataset.decode(result)


def main():
    print("=== Tiny Transformer Inference ===")

    # 1. Init
    dataset = AdditionDataset(max_digits=config.MAX_DIGITS)

    # Calculate sufficient max_len just like in train.py
    max_seq_len = dataset.max_input_len + dataset.max_output_len + 10

    model = Transformer(
        src_vocab_size=dataset.vocab_size,
        tgt_vocab_size=dataset.vocab_size,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        d_ff=config.D_FF,
        max_len=max_seq_len,
        dropout=0.0,
    )

    # 2. Load
    model.load_weights(config.MODEL_FILENAME)

    print("Type a 2-digit addition problem (e.g. '12+34') or 'q' to quit.")

    while True:
        text = input("\nProblem > ").strip().replace(" ", "")
        if text.lower() == "q":
            break

        if "+" not in text:
            print("Please use format 'A+B'")
            continue

        try:
            # Predict
            output = greedy_decode(model, dataset, text)
            print(f"Model   > {output}")

            # Verify
            a, b = map(int, text.split("+"))
            correct = str(a + b)
            mark = "✅" if output == correct else f"❌ ({correct})"
            print(f"Check   > {mark}")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
