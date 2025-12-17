"""OpenAI / OpenAI 兼容 API 客户端适配器。

说明：
- 使用 `openai` 官方新版本 SDK（>=1.0），通过 `OpenAI` 客户端调用。
- 支持从 .env 读取 API KEY 和 BASE_URL。
- 只实现了最基础的 `generate`，用于单轮文本生成。
"""

from __future__ import annotations

import os
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.core.llm_base import BaseLLM, LLMResult


class OpenAIClient(BaseLLM):
  """简单的 OpenAI 文本生成客户端。

  你可以把它理解为：
  - 上层 Solver 不关心具体是哪个模型
  - 只要这个类实现了 `generate`，上层就能统一调用
  """

  def __init__(
    self,
    model_name: str,
    base_url: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
  ) -> None:
    # 加载 .env 文件中的环境变量
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
      raise ValueError("OPENAI_API_KEY 未配置，请在项目根目录的 .env 文件中设置。")

    # 如果你使用的是 OpenAI 兼容的私有部署，可以在 .env 中修改 OPENAI_BASE_URL
    base_url = base_url or os.getenv("OPENAI_BASE_URL") or None

    self.client = OpenAI(api_key=api_key, base_url=base_url)
    self.model_name = model_name
    self.max_tokens = max_tokens
    self.temperature = temperature

  def generate(self, prompt: str, **kwargs: Any) -> LLMResult:
    """调用 OpenAI 的 chat.completions 接口进行生成。

    为了简单起见，我们把所有任务都包装成单轮对话：
    - system: 简短说明
    - user: 传入的 prompt
    """

    # 允许调用时临时覆盖 max_tokens / temperature
    max_tokens = kwargs.get("max_tokens", self.max_tokens)
    temperature = kwargs.get("temperature", self.temperature)

    response = self.client.chat.completions.create(
      model=self.model_name,
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
      ],
      max_tokens=max_tokens,
      temperature=temperature,
    )

    choice = response.choices[0]
    text = choice.message.content or ""

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    return LLMResult(
      text=text,
      prompt_tokens=prompt_tokens,
      completion_tokens=completion_tokens,
      total_tokens=total_tokens,
      raw=response.to_dict() if hasattr(response, "to_dict") else dict(response),
    )
