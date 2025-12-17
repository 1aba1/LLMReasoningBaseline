"""Direct 输出方法实现。

核心思想：
- 不强制模型展示详细推理过程
- 直接让模型输出最终选项（A/B/C/D）
"""

from __future__ import annotations

from typing import Dict, Any

from src.core.solver_base import SolverBase
from src.core.llm_base import LLMResult
from src.utils.answer_extractor import extract_choice_answer


class DirectSolver(SolverBase):
  """最基础的 Direct 输出推理方法。"""

  def build_prompt(self, sample: Dict[str, Any]) -> str:
    # 这里假设 prompt_template 中使用 {question} 作为占位符
    question = sample.get("question", "")
    return self.prompt_template.format(question=question)

  def parse_answer(self, llm_result: LLMResult) -> str:
    # 使用工具函数，从模型输出中提取 A/B/C/D
    return extract_choice_answer(llm_result.text)
