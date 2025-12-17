"""从模型输出中提取选择题答案的工具。

目前假设题目的答案是 A/B/C/D 中的一个大写字母：
- 如果模型输出中包含多个大写字母，只取第一个匹配的
- 如果没有找到有效选项，返回 ""（空字符串），方便后续统计为错误
"""

from __future__ import annotations

import re


CHOICE_PATTERN = re.compile(r"[ABCD]")


def extract_choice_answer(text: str) -> str:
  """从文本中提取第一个 A/B/C/D 选项。"""

  if not text:
    return ""

  match = CHOICE_PATTERN.search(text)
  if match:
    return match.group(0)
  return ""
