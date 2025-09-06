import os
import random
import numpy as np
import tensorflow as tf

def set_seed(s: int = 42):
    """
    Set seeds for reproducibility across Python, NumPy, and TensorFlow.

    Args:
        s: Seed integer.
    """
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
    
    # Optional: force deterministic GPU operations (slower)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        # For older TF versions where this isn't available
        pass
