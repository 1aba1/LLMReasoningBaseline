"""Token 统计工具（占位，当前主要依赖模型返回的 usage）。

说明：
- 很多 LLM API（包括 OpenAI）都会在返回结果里包含 token 使用情况（usage 字段），
  本项目的主流程里已经直接读取了这些字段。
- 如果你后续使用的是本地模型，或者自建服务没有返回 usage，
  可以在这里接入第三方库（如 tiktoken）来估算 token 数。
"""

from __future__ import annotations

from typing import Iterable


def sum_tokens(usages: Iterable[dict]) -> dict:
  """对一组 usage 字典求和，得到总的 token 统计。

  输入示例：
  - [{"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}, ...]

  返回示例：
  - {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}
  """

  total_prompt = 0
  total_completion = 0
  total_total = 0

  for u in usages:
    if not u:
      continue
    total_prompt += int(u.get("prompt_tokens") or 0)
    total_completion += int(u.get("completion_tokens") or 0)
    total_total += int(u.get("total_tokens") or 0)

  return {
    "prompt_tokens": total_prompt,
    "completion_tokens": total_completion,
    "total_tokens": total_total,
  }
