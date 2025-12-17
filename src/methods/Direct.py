"""Direct 输出方法实现。
- 直接让模型输出最终选项（A/B/C/D）
- 单轮推理：一次 prompt → 一次 completion → 结束
"""

from typing import Dict, Any

from src.core.solver_base import SolverBase, SolveResult
from src.utils.answer_extractor import extract_choice_answer


class DirectSolver(SolverBase):
  def solve(self, sample: Dict[str, Any]) -> SolveResult:
    """执行 Direct 推理：单轮 LLM 调用。

    实现步骤：
    1. 根据样本构造 prompt
    2. 调用 LLM 生成回复
    3. 从回复中解析出最终答案（A/B/C/D）
    """
    # 构造 prompt
    question = sample.get("question", "")  #从输入的题目数据里，取出「问题内容」（如果没有就取空字符串）
    prompt = self.prompt_template.format(question=question)

    # 调用 LLM（单轮调用）
    llm_result = self.llm.generate(prompt)

    # 解析最终答案
    final_answer = extract_choice_answer(llm_result.text)

    # 返回 SolveResult（单轮推理，所以 llm_results 只包含一个结果）
    return SolveResult(
      final_answer=final_answer,
      raw_output=llm_result.text,
      raw_input=prompt,
      llm_results=[llm_result],
    )
