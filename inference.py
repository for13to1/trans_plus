import numpy as np
from autograd import Tensor
from dataset import AdditionDataset
from model import TransformerModel

def greedy_decode(model, dataset, problem_str, max_len=5):
    # Encode Input
    # "12+34" -> [1, 2, +, 3, 4]
    src_ids = dataset.encode(problem_str)

    # We need to feed [src] + [<sos>] to start generating
    # Current Model Logic (train.py):
    # It takes a full sequence `src + tgt` and predicts next token.
    # We can simulate generation by appending predicted token one by one.

    # Start: "12+34" + "<sos>"
    # Pad src to max length?
    # The Dataset padding logic puts padding at end.
    # Our simple inference should match what model saw during training.
    # Training saw: "12+34 <pad> <pad> <sos> 46 <eos> <pad>"

    # Let's constructing explicit input:
    # "12+34"
    # Append <sos>

    # Pad src to max length to match training distribution
    pad_len = dataset.max_input_len - len(src_ids)
    if pad_len > 0:
        src_ids = src_ids + [dataset.pad_id] * pad_len

    current_input = src_ids + [dataset.sos_id]

    result = []

    for _ in range(max_len):
        # Prepare input tensor
        # Add batch dim: (1, L)
        input_ids = np.array([current_input])

        # Mask
        # We need both Causal Mask AND Padding Mask
        L = input_ids.shape[1]

        # 1. Causal
        causal_mask_val = np.triu(np.ones((L, L)), k=1) * -1e9

        # 2. Padding
        is_pad = (input_ids == dataset.pad_id) # (1, L)
        pad_mask_val = is_pad[:, np.newaxis, np.newaxis, :] * -1e9

        # Combine
        combined_mask_val = causal_mask_val[np.newaxis, np.newaxis, :, :] + pad_mask_val

        mask = Tensor(combined_mask_val)

        # Forward
        logits = model(input_ids, mask) # (1, L, V)

        # Get last token prediction
        last_logits = logits.data[0, -1, :]
        next_id = np.argmax(last_logits)

        if next_id == dataset.eos_id:
            break

        result.append(next_id)
        current_input.append(next_id)

    return dataset.decode(result)

def main():
    print("=== Tiny Transformer Inference ===")

    # 1. Init
    dataset = AdditionDataset(max_digits=2)
    d_model = 48
    model = TransformerModel(
        vocab_size=dataset.vocab_size,
        d_model=d_model,
        num_heads=4,
        num_layers=2
    )

    # 2. Load
    model.load_weights('tiny_model.pkl')

    print("Type a 2-digit addition problem (e.g. '12+34') or 'q' to quit.")

    while True:
        text = input("\nProblem > ").strip().replace(" ", "")
        if text.lower() == 'q':
            break

        if '+' not in text:
            print("Please use format 'A+B'")
            continue

        try:
            # Predict
            output = greedy_decode(model, dataset, text)
            print(f"Model   > {output}")

            # Verify
            a, b = map(int, text.split('+'))
            correct = str(a + b)
            mark = "✅" if output == correct else f"❌ ({correct})"
            print(f"Check   > {mark}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
