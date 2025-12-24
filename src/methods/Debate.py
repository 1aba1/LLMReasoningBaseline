"""Multi-agent debate 推理范式实现。

特点：
- 至少三个角色轮流发言（默认：Proponent、Skeptic、Judge）
- 每轮携带完整辩论历史，便于上下文延续
- 当裁判角色给出 <final_answer>（或解析出 A/B/C/D）即终止
"""

import re
from typing import Any, Dict, List, Optional

from src.core.solver_base import SolverBase, SolveResult
from src.utils.answer_extractor import extract_choice_answer


class DebateSolver(SolverBase):
  def __init__(
    self,
    llm,
    prompt_template: str,
    max_rounds: int = 3,
    roles: Optional[List[Dict[str, str]]] = None,
  ) -> None:
    """初始化多智能体辩论 Solver。

    参数
    ------
    max_rounds: int
      最多辩论轮数（每轮包含所有角色各一次发言）
    roles: Optional[List[Dict[str, str]]]
      角色定义列表，元素至少包含 name 与 goal；如果未传入则使用默认三角色。
    """
    super().__init__(llm=llm, prompt_template=prompt_template)
    self.max_rounds = max_rounds
    self.roles = roles or [
      {"name": "Proponent", "goal": "提出系统化解题思路，给出可行解法与推导，并推导出答案"},
      {"name": "Skeptic", "goal": "审视并质疑推理步骤，寻找漏洞与替代思路，并推导出答案"},
      {"name": "Judge", "goal": "综合双方观点，并在推理信息充分时给出<final_answer>。"},
    ]

  def solve(self, sample: Dict[str, Any]) -> SolveResult:
    question = sample.get("question", "")
    history: List[Dict[str, Any]] = []
    llm_results = []
    raw_outputs: List[str] = []
    final_answer: str = ""
    last_prompt = ""

    for round_idx in range(1, self.max_rounds + 1):
      for agent in self.roles:
        history_text = self._format_history(history)
        prompt = self.prompt_template.format(
          role_name=agent["name"],
          role_goal=agent["goal"],
          debate_history=history_text if history_text else "暂无历史，作为首轮发言。",
          question=question,
        )
        last_prompt = prompt

        llm_result = self.llm.generate(prompt)
        llm_results.append(llm_result)

        output_text = (llm_result.text or "").strip()
        raw_outputs.append(f"[Round {round_idx}][{agent['name']}]\n{output_text}")

        extracted = self._extract_final_answer(output_text)
        history.append(
          {
            "round": round_idx,
            "agent": agent["name"],
            "role_goal": agent["goal"],
            "prompt": prompt,
            "raw_response": output_text,
            "final_answer": extracted or "",
          }
        )

        # 只有Judge角色的final_answer才能终止辩论
        if extracted and agent["name"] == "Judge":
          final_answer = extracted
          break

      if final_answer:
        break

    # 兜底：如果未找到答案，尝试从最后一条回复中解析
    if not final_answer and raw_outputs:
      final_answer = self._extract_final_answer(raw_outputs[-1]) or extract_choice_answer(
        raw_outputs[-1]
      )

    return SolveResult(
      final_answer=final_answer,
      raw_output="\n\n".join(raw_outputs),
      raw_input=last_prompt,
      llm_results=llm_results,
      intermediate_steps=history,
    )

  @staticmethod
  def _format_history(history: List[Dict[str, Any]]) -> str:
    """将历史记录格式化为可读文本供下一轮提示使用。"""
    lines = []
    for step in history:
      lines.append(
        f"[Round {step['round']}][{step['agent']}] {step['raw_response']}"
      )
    return "\n".join(lines)

  @staticmethod
  def _extract_final_answer(text: str) -> str:
    """优先解析 <final_answer> 标签，其次回退到选项提取。"""
    pattern = r"<final_answer>\s*(.*?)\s*</final_answer>"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if m:
      inner = m.group(1).strip()
      return extract_choice_answer(inner) or inner
    return extract_choice_answer(text)

