"""评估指标工具：正确率 & token 统计。"""

from __future__ import annotations

from typing import List, Dict, Any

from src.utils.token_counter import sum_tokens


def compute_accuracy(records: List[Dict[str, Any]]) -> float:
  """根据一组记录计算选择题的准确率。

  每条记录需包含：
  - gold_answer: 标准答案（A/B/C/D）
  - pred_answer: 模型预测答案（A/B/C/D 或空字符串）
  """

  if not records:
    return 0.0

  correct = 0
  total = 0
  for r in records:
    gold = (r.get("gold_answer") or "").strip().upper()
    pred = (r.get("pred_answer") or "").strip().upper()
    if not gold:
      # 如果没有标准答案，跳过该样本
      continue
    total += 1
    if gold == pred:
      correct += 1

  if total == 0:
    return 0.0
  return correct / total


def aggregate_token_usage(records: List[Dict[str, Any]]) -> Dict[str, int]:
  """汇总一组记录中的 token 使用情况。"""

  usages = [r.get("usage") or {} for r in records]
  return sum_tokens(usages)
