"""Solver 工厂：根据 method 配置创建对应的推理 Solver。

设计目标：
- 像 `choose_llm` 一样，把「根据 method_name 选择 Solver 并初始化」的逻辑集中在这里
- `main.py` 只关心高层流程，不再关心具体有哪些 Solver
- 后续新增 Solver（比如 `SelfConsistencySolver` 等）只需要在这里注册
"""

from typing import Any, Dict, Type

from src.core.llm_base import BaseLLM
from src.core.solver_base import SolverBase
from src.methods.CoT import CoTSolver
from src.methods.Debate import DebateSolver
from src.methods.React import ReactSolver
from src.methods.Direct import DirectSolver


# 方法注册表（给方法起「别名」，方便调用和扩展）
METHOD_REGISTRY: Dict[str, Type[SolverBase]] = {
  "Direct": DirectSolver,
  "CoT": CoTSolver,
  "React": ReactSolver,
  "Debate": DebateSolver,
  # 未来新增的推理范式，可以在这里继续注册：
  # "self_consistency": SelfConsistencySolver,
}


def create_solver(method_cfg: Dict[str, Any], llm: BaseLLM, prompt_template: str) -> SolverBase:
  """根据 method 配置创建对应的 Solver 实例。

  参数
  ------
  method_cfg: Dict[str, Any]
    方法配置字典，至少包含 `method_name`
  llm: BaseLLM
    已经初始化好的 LLM 客户端
  prompt_template: str
    本次实验使用的 prompt 模板内容
  """
  method_name = method_cfg.get("method_name")
  solver_cls = METHOD_REGISTRY.get(method_name)
  if solver_cls is None:
    raise ValueError(f"不支持的 method_name: {method_name}，可选值为: {list(METHOD_REGISTRY.keys())}")

  return solver_cls(llm=llm, prompt_template=prompt_template)

