"""OpenAI客户端适配器。

说明：
- 使用 `openai` 官方新版本 SDK（>=1.0），通过 `OpenAI` 客户端调用。
- 支持从 .env 读取 API KEY
- 用于单轮文本生成。
"""

from typing import Any, Optional
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
    base_url: Optional[str],
    max_tokens: int ,
    temperature: float ,
    api_key:str
  ) -> None:

    self.api_key = api_key
    if not api_key:
      raise ValueError("OPENAI_API_KEY 未配置，请在项目根目录的 .env 文件中设置。")

    self.client = OpenAI(api_key=api_key, base_url=base_url)
    self.model_name = model_name
    self.max_tokens = max_tokens
    self.temperature = temperature

  def generate(self, prompt: str, **kwargs: Any) -> LLMResult:
    """调用 OpenAI 的 chat.completions 接口进行生成。

    - system: 简短说明
    - user: 传入的 prompt
    """

    # 允许调用时临时覆盖 max_tokens / temperature
    # max_tokens = kwargs.get("max_tokens", self.max_tokens)
    # temperature = kwargs.get("temperature", self.temperature)

    response = self.client.chat.completions.create(
      model=self.model_name,
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
      ],
      max_tokens=self.max_tokens,
      temperature=self.temperature,
    )

    choice = response.choices[0]
    text = choice.message.content or ""

    # getattr 是 Python 内置的函数，专门用来从 “对象” 里拿指定的 “属性"
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    # 返回openai客户端一次调用的结果
    return LLMResult(
      text=text,
      prompt_tokens=prompt_tokens,
      completion_tokens=completion_tokens,
      total_tokens=total_tokens,
      raw=response.to_dict() if hasattr(response, "to_dict") else dict(response),
    )
