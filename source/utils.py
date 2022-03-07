import numpy as np
import time
from typing import Tuple

def get_arrays_size_MB (arrays: Tuple[np.array]):
    return sum([arr.nbytes for arr in arrays]) / 1024 / 1024

def millis():
    return round(time.time() * 1000)
