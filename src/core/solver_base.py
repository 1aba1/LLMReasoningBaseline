"""推理范式（方法）的抽象基类。

Direct、CoT、ReAct 等不同的推理范式，统一继承自这个基类。
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

from src.core.llm_base import BaseLLM, LLMResult


class SolverBase(ABC):
  """推理范式的基类。

  一个 Solver 负责：
  - 如何根据一个样本（题目）构造 prompt
  - 如何调用 LLM
  - 如何从 LLM 输出中提取最终答案
  """

  def __init__(self, llm: BaseLLM, prompt_template: str):
    self.llm = llm
    self.prompt_template = prompt_template

  @abstractmethod
  def build_prompt(self, sample: Dict[str, Any]) -> str:
    """根据数据样本构造 prompt。

    默认假设样本中至少包含字段：
    - "question": 题目文本
    """

    raise NotImplementedError

  @abstractmethod
  def parse_answer(self, llm_result: LLMResult) -> str:
    """从 LLM 输出中解析出最终答案（例如 A/B/C/D）。"""

    raise NotImplementedError

  def run_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """对单个样本运行推理流程。
    """

    prompt = self.build_prompt(sample)
    llm_result = self.llm.generate(prompt)
    pred_answer = self.parse_answer(llm_result)

    return {
      "id": sample.get("id"),
      "question": sample.get("question"),
      "gold_answer": sample.get("answer"),
      "pred_answer": pred_answer,
      "raw_output": llm_result.text,
      "usage": {
        "prompt_tokens": llm_result.prompt_tokens,
        "completion_tokens": llm_result.completion_tokens,
        "total_tokens": llm_result.total_tokens,
      },
    }
