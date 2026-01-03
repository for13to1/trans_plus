# Configuration
import os

# Dataset Config
MAX_DIGITS = 2

# Model Config
D_MODEL = 48
NUM_HEADS = 4
NUM_LAYERS = 2

# Training Config
BATCH_SIZE = 32
MAX_STEPS = 3000
LEARNING_RATE = 0.001

# File Config
MODEL_FILENAME = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "addition_model.pkl"
)
