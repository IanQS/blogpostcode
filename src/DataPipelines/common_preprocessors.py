"""
common_preprocessors
    - simple example of a preprocessor
    - called right before the provide() method in data_backend

Author: Ian Q.

Notes:
"""

import numpy as np

def log_normalizations(incoming_data):
    return np.log1p(incoming_data / incoming_data[0])