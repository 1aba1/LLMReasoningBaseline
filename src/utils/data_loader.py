"""数据加载工具：按照最大样本数读取 JSON 数据集。"""
import json
from typing import List, Dict, Any, Optional


def load_json_dataset(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
  """从给定路径加载 JSON 格式的数据集。

  要求文件内容是一个列表，每个元素是一个字典，例如：
  [{"id": 1, "question": "...", "answer": "B"}, ...]
  """

  with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

  if not isinstance(data, list):
    raise ValueError(f"数据文件 {path} 的顶层结构必须是列表（list）。")

  if max_samples is not None:
    data = data[: int(max_samples)]

  return data
