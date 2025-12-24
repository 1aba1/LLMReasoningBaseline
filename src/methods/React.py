"""ReAct 推理范式实现。

核心特点：
- 维护多轮 thought/action/observation 历史，并在每轮 prompt 中显式提供。
- 当检测到 <final_answer>（或解析出 A/B/C/D）时终止。
"""

import re
from typing import Any, Dict, List, Optional

from src.core.solver_base import SolverBase, SolveResult
from src.utils.answer_extractor import extract_choice_answer


class ReactSolver(SolverBase):
  def __init__(self, llm, prompt_template: str, max_rounds: int = 8) -> None:
    super().__init__(llm=llm, prompt_template=prompt_template)
    self.max_rounds = max_rounds

  def solve(self, sample: Dict[str, Any]) -> SolveResult:
    question = sample.get("question", "")
    history: List[Dict[str, Any]] = []
    llm_results = []
    final_answer: str = ""
    raw_outputs: List[str] = []
    last_prompt = ""

    for round_idx in range(1, self.max_rounds + 1):
      history_text = "\n".join(step["raw_response"] for step in history) if history else ""
      prompt = self._build_prompt(question=question, history_text=history_text)
      last_prompt = prompt

      llm_result = self.llm.generate(prompt)
      llm_results.append(llm_result)

      output_text = (llm_result.text or "").strip()
      raw_outputs.append(output_text)

      parsed = self._extract_tags(output_text)

      # 记录中间步骤，便于日志与下轮构造
      history.append(
        {
          "round": round_idx,
          "prompt": prompt,
          "raw_response": output_text,
          "thought": parsed.get("thought"),
          "action": parsed.get("action"),
          "observation": parsed.get("observation"),
          "final_answer": parsed.get("final_answer"),
        }
      )

      # 终止条件：显式 final_answer 或解析出的选项
      if parsed.get("final_answer"):
        final_answer = parsed["final_answer"]
        break

      extracted_choice = extract_choice_answer(output_text)
      if extracted_choice:
        final_answer = extracted_choice
        break

    # 如果循环结束仍无答案，兜底用最后一轮尝试解析
    if not final_answer and raw_outputs:
      final_answer = extract_choice_answer(raw_outputs[-1])

    return SolveResult(
      final_answer=final_answer,
      raw_output="\n\n".join(raw_outputs),
      raw_input=last_prompt,
      llm_results=llm_results,
      intermediate_steps=history,
    )

  def _build_prompt(self, question: str, history_text: str) -> str:
    """将模板与历史拼接成本轮 prompt。"""
    base_prompt = self.prompt_template.format(question=question)
    history_block = history_text if history_text else "（暂无历史，直接开始第 1 轮推理）"
    continuation_hint = (
      "请在历史基础上继续输出下一轮的<thought>与<action>/<final_answer>；"
      "若输出<action>，务必给出对应的<observation>。"
    )
    return f"{base_prompt}\n\n历史推理回顾：\n{history_block}\n\n{continuation_hint}"

  @staticmethod
  def _extract_tags(text: str) -> Dict[str, Optional[str]]:
    """简单解析模型输出中的 ReAct 标签。"""
    def _find(tag: str) -> Optional[str]:
      pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
      m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
      return m.group(1).strip() if m else None

    result = {
      "thought": _find("thought"),
      "action": _find("action"),
      "observation": _find("observation"),
      "final_answer": None,
    }

    final_answer = _find("final_answer")
    if final_answer:
      cleaned = extract_choice_answer(final_answer) or final_answer.strip()
      result["final_answer"] = cleaned

    return result
