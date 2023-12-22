import numpy as np


def np_to_cpp_array(arr: np.ndarray) -> str:
    array_str = ", ".join(map(str, arr.flatten()))
    return f"{{ {array_str} }}"
