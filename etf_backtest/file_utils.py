"""源码和模型文件共用的内容摘要。"""
import hashlib
from pathlib import Path


# 读取文件内容并计算 SHA-256，供策略、资源和模型产物身份核对。
def sha256_file(path: Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
