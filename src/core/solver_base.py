"""推理范式（方法）的抽象基类。

Direct、CoT、ReAct、Multi-agent-debate 等不同的推理范式，统一继承自这个基类。

设计说明：
- `solve` 方法：子类必须实现，负责核心的推理逻辑（可以是单轮或多轮）
- `run_sample` 方法：模板方法，负责样本级别的包装（输入输出格式化、token统计等）
- 这样设计使得 Direct（单轮）和 ReAct/Debate（多轮）都能很好地适配
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from src.core.llm_base import BaseLLM, LLMResult


@dataclass
class SolveResult:
  """推理范式结果的结构化表示。
  用于封装 solve 方法的返回结果，支持单轮和多轮推理范式。
  """

  final_answer: str  # 最终答案（例如 A/B/C/D）
  raw_output: str  # 模型输出的原始文本（单轮时为一个字符串，多轮时可以是所有轮次的文本拼接或最后一次的输出）
  raw_input: str  # 发送给大模型的完整输入（包括问题与提示词）
  llm_results: List[LLMResult]  # 所有 LLM 调用结果的列表（多轮推理时会包含多次调用的结果）
  intermediate_steps: Optional[List[Dict[str, Any]]] = None   # 可选的中间步骤信息（用于 ReAct、Debate 等需要展示推理过程的范式）


class SolverBase(ABC):
  """推理范式的基类。

  一个 Solver 负责：
  - 如何对给定样本进行推理（可能涉及单轮或多轮 LLM 调用）
  - 如何从推理结果中提取最终答案
  - 如何管理推理过程中的状态（如对话历史、agent状态等）

  设计要点：
  - `solve` 方法是核心抽象方法，子类实现自己的推理逻辑
  - Direct 范式：solve 中调用一次 LLM 即可
  - ReAct 范式：solve 中循环调用 LLM，直到满足终止条件
  - Multi-agent-debate：solve 中管理多个 agent 的对话，循环直到收敛或达到最大轮次
  """

  def __init__(self, llm: BaseLLM, prompt_template: str):
    self.llm = llm
    self.prompt_template = prompt_template

  @abstractmethod
  def solve(self, sample: Dict[str, Any]) -> SolveResult:
    """执行推理过程的核心方法。

    这个方法应该包含完整的推理逻辑，可以：
    - 单轮推理：调用一次 LLM 并返回结果（如 Direct、CoT）
    - 多轮推理：循环调用 LLM，维护状态，直到满足终止条件（如 ReAct、Debate）

    参数
    ------
    sample: Dict[str, Any]
      数据样本，至少包含 "question" 字段

    返回
    ------
    SolveResult
      包含最终答案、原始输出、所有 LLM 调用结果等信息

    实现提示：
    - Direct/CoT：构造 prompt → 调用一次 self.llm.generate → 解析答案 → 返回 SolveResult
    - ReAct：循环执行（构造 prompt → 调用 LLM → 解析 action/observation → 判断是否终止）
    - Debate：管理多个 agent，循环执行多轮对话，直到收敛或达到最大轮次
    """
    raise NotImplementedError

  def run_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """对单个样本运行推理流程（模板方法）。

    这个方法负责：
    1. 调用子类实现的 solve 方法执行推理
    2. 格式化返回结果，包含样本信息、预测答案、token 使用等
    3. 聚合多轮调用的 token 统计信息

    参数
    ------
    sample: Dict[str, Any]
      数据样本

    返回
    ------
    Dict[str, Any]
      包含以下字段：
      - id: 样本 ID
      - question: 题目文本
      - gold_answer: 标准答案
      - pred_answer: 预测答案
      - raw_input: 发送给大模型的完整输入（包括问题与提示词）
      - raw_output: 模型原始输出
      - usage: token 使用统计（聚合所有轮次）
      - intermediate_steps: 可选的中间步骤信息（多轮推理时有用）
    """
    solve_result = self.solve(sample)

    # 聚合所有 LLM 调用的 token 使用情况
    total_prompt_tokens = sum(
      r.prompt_tokens or 0 for r in solve_result.llm_results
    )
    total_completion_tokens = sum(
      r.completion_tokens or 0 for r in solve_result.llm_results
    )
    total_tokens = sum(
      r.total_tokens or 0 for r in solve_result.llm_results
    )

    result = {
      "id": sample.get("id"),
      "question": sample.get("question"),
      "gold_answer": sample.get("answer"),
      "pred_answer": solve_result.final_answer,
      "raw_input": solve_result.raw_input,
      "raw_output": solve_result.raw_output,
      "usage": {
        "prompt_tokens": total_prompt_tokens if total_prompt_tokens > 0 else None,
        "completion_tokens": total_completion_tokens if total_completion_tokens > 0 else None,
        "total_tokens": total_tokens if total_tokens > 0 else None,
      },
    }

    # 如果有多轮推理的中间步骤信息，添加到结果中
    if solve_result.intermediate_steps:
      result["intermediate_steps"] = solve_result.intermediate_steps

    return result
