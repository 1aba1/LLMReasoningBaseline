"""定义LLM 抽象基类 与 LLM 调用的结构化结果

LLM 抽象基类使得后续无论是调用 OpenAI、DeepSeek
还是你们实验室自建的大模型服务，都可以通过继承这个基类来适配。
"""

from abc import ABC, abstractmethod  # 用来定义“抽象类”
from dataclasses import dataclass   # 用来快速定义“结构化数据”
from typing import Optional, Dict, Any # 类型注解


@dataclass # # 这个装饰器让下面的类变成“数据打包类”
class LLMResult:
  """LLM 调用的结构化结果。

  - text:       模型输出的完整文本
  - prompt_tokens:  提示词 token 数
  - completion_tokens:  输出 token 数
  - total_tokens:  总 token 数
  - raw:        底层 API 返回的原始字典，方便调试
  """

  text: str
  prompt_tokens: Optional[int] = None
  completion_tokens: Optional[int] = None
  total_tokens: Optional[int] = None
  raw: Optional[Dict[str, Any]] = None


class BaseLLM(ABC):
  """LLM 抽象基类

  所有具体的大模型客户端（OpenAI、本地模型等）都应该继承这个类，
  并实现 `generate` 方法。
  """

  @abstractmethod #装饰器：标记这个方法是“必须实现的规则”
  def generate(self, prompt: str, **kwargs: Any) -> LLMResult:
    """给定一个 prompt，调用底层模型并返回结构化结果。

    参数
    ------
    prompt: str
      完整的文本提示词

    返回
    ------
    LLMResult
      包含模型输出文本和 token 使用信息
    """

    raise NotImplementedError #若没有重写。主动抛出异常，明确提示 “这个方法没实现”
