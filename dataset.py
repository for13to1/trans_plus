import numpy as np
import random

# SOS: Start of Sequence
# EOS: End of Sequence
# PAD: Padding

class AdditionDataset:
    """
    Dataset for generating N-digit addition problems.
    Input: "123+456" (as tokens)
    Output: "579" (as tokens)
    """

    def __init__(self, max_digits=3):
        self.max_digits = max_digits

        # Build Vocabulary
        # 0-9, +, <sos>, <eos>, <pad>
        self.chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ', '<sos>', '<eos>', '<pad>']
        self.char_to_id = {c: i for i, c in enumerate(self.chars)}
        self.id_to_char = {i: c for i, c in enumerate(self.chars)}

        self.pad_id = self.char_to_id['<pad>']
        self.sos_id = self.char_to_id['<sos>']
        self.eos_id = self.char_to_id['<eos>']
        self.vocab_size = len(self.chars)

        # Calculate max length
        self.max_input_len = max_digits * 2 + 1
        self.max_output_len = max_digits + 1 + 2

    def encode(self, text):
        return [self.char_to_id[c] for c in text]

    def decode(self, ids):
        return "".join([self.id_to_char[i] for i in ids if i not in [self.pad_id, self.sos_id, self.eos_id]])

    def get_batch(self, batch_size=32):
        src_batch = []
        tgt_batch = []

        for _ in range(batch_size):
            n1 = random.randint(0, 10**self.max_digits - 1)
            n2 = random.randint(0, 10**self.max_digits - 1)

            problem_str = f"{n1}+{n2}"
            ans_str = str(n1 + n2)

            src = self.encode(problem_str)
            tgt = [self.sos_id] + self.encode(ans_str) + [self.eos_id]

            # Pad
            src = src + [self.pad_id] * (self.max_input_len - len(src))
            tgt = tgt + [self.pad_id] * (self.max_output_len - len(tgt))

            src_batch.append(src)
            tgt_batch.append(tgt)

        return np.array(src_batch), np.array(tgt_batch)
