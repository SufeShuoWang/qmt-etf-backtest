"""规则无关的模型设备名称校验；导入时不加载任何训练后端。"""

import re


def normalize_model_device(value: object) -> str:
    """接受 cpu、cuda、cuda:N，并将 gpu／gpu:N 别名规范为 CUDA 设备。"""
    if not isinstance(value, str):
        raise TypeError("model device must be a string")
    device = value.strip().lower()
    if device == "gpu" or device.startswith("gpu:"):
        device = "cuda" + device[3:]
    if device != "cpu" and re.fullmatch(r"cuda(?::(?:0|[1-9][0-9]*))?", device) is None:
        raise ValueError("model device must be cpu, cuda, or cuda:N (GPU index)")
    return device
