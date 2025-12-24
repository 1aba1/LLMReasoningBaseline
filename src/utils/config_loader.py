"""配置和文件加载工具：读取 YAML 配置文件和文本文件。"""
import yaml
from typing import Any, Dict


def load_yaml(path: str) -> Dict[str, Any]:
  """读取 YAML 配置文件,并返回字典格式的配置数据"""
  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def read_text(path: str) -> str:
  """读取纯文本文件（如 prompt 模板）。"""
  with open(path, "r", encoding="utf-8") as f:
    return f.read()
