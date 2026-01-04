import numpy as np


def print_header(title):
    print("╔" + "═" * 58 + "╗")
    print(f"║ {title.center(56)} ║")
    print("╚" + "═" * 58 + "╝")


def analyze_parameters(model):
    """
    Analyzes and prints the parameter distribution of a Transformer model.
    """
    print_header("MODEL ANALYSIS REPORT")

    # 1. Component Analysis
    print(" [1] PARAMETER COUNTS")

    if hasattr(model, "shared_weight"):
        shared_weight = model.shared_weight.data.size
        print(f"     Shared Weights: ...... {shared_weight:,}")
    else:
        print("     Shared Weights: ...... None")

    if hasattr(model, "encoder"):
        encoder_params = sum(p.data.size for p in model.encoder.parameters())
        print(f"     Encoder: ............. {encoder_params:,}")

    if hasattr(model, "decoder"):
        decoder_params = sum(p.data.size for p in model.decoder.parameters())
        print(f"     Decoder: ............. {decoder_params:,}")

    # 2. Total Analysis
    all_params = model.parameters()
    total_params = sum(p.data.size for p in all_params)
    print("     " + "-" * 30)
    print(f"     TOTAL: ............... {total_params:,}")

    # 3. Duplicate Check
    param_ids = set(id(p.data) for p in all_params)
    unique_count = len(param_ids)
    total_obj_count = len(all_params)

    print("\n [2] STRUCTURE CHECK")
    if unique_count != total_obj_count:
        print(
            f"     Unique Params: ....... {unique_count}/{total_obj_count} (Shared detected)"
        )
    else:
        print(
            f"     Unique Params: ....... {unique_count}/{total_obj_count} (No shared weights)"
        )


def analyze_memory_usage(d_model, d_ff, num_layers, vocab_size, num_heads=8):
    """
    Prints a theoretical memory usage estimation based on model hyperparameters.
    """
    print("\n [3] MEMORY ESTIMATION (Theoretical FP32)")

    # Theoretical Calculations
    enc_total = (4 * d_model**2) + (2 * d_model * d_ff) + (2 * d_model)
    dec_total = (8 * d_model**2) + (2 * d_model * d_ff) + (3 * d_model)

    embed_params = vocab_size * d_model
    total_theoretical = embed_params + num_layers * (enc_total + dec_total)

    # Memory Calc
    param_bytes = total_theoretical * 4  # FP32
    train_bytes = param_bytes * 4  # Weights + Gradients + Optimizer(m, v)

    print(f"     Weights Only: ........ {param_bytes / 1024 / 1024:.2f} MB")
    print(
        f"     Training Overhead: ... ~{train_bytes / 1024 / 1024:.2f} MB (Weights+Grads+Opt)"
    )
    print("=" * 60 + "\n")
