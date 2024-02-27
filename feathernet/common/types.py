from typing import Literal

import numpy as np

DeviceType = Literal["cpu", "gpu"]
ShapeType = tuple[int, ...]
TensorSupported = int | float | np.ndarray
