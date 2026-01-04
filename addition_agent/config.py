# Configuration
import os

# Dataset Config
MAX_DIGITS = 2

# Model Config - Tiny Transformer for CPU Autograd Training
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 512

# Training Config
BATCH_SIZE = 64
MAX_STEPS = 3000
LEARNING_RATE = 0.0  # Handled by Noam scheduler
WARMUP_STEPS = 1000
LABEL_SMOOTHING = 0.1

# File Config
MODEL_FILENAME = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "addition_model.pkl"
)
