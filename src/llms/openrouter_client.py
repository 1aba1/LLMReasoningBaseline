"""OpenRouter 客户端适配器。

说明：
- 通过 OpenRouter 聚合多种后端模型（OpenAI、DeepSeek、Qwen 等）
- 使用与 OpenAIClient 相同的 BaseLLM 接口，便于在配置中自由切换
"""

from typing import Any, Optional, Dict
from openai import OpenAI

from src.core.llm_base import BaseLLM, LLMResult


class OpenRouterClient(BaseLLM):
  """基于 OpenRouter 的文本生成客户端。

  """

  def __init__(
    self,
    model_name: str,
    base_url: Optional[str],
    max_tokens: int,
    temperature: float,
    api_key: str,
    extra_headers: Optional[Dict[str, str]] = None,
  ) -> None:

    if not api_key:
      raise ValueError(
        "OPENROUTER_API_KEY 未配置，请在项目根目录的 .env 文件中设置。"
      )

    # OpenRouter 官方推荐附带 Referer 和 X-Title，但不是强制的
    default_headers: Dict[str, str] = {
      "HTTP-Referer": "http://localhost",  # 可按需在配置中改成你的服务地址
      "X-Title": "Reasoning-Framework-Experiment",
    }
    if extra_headers:
      default_headers.update(extra_headers)

    self.client = OpenAI(
      api_key=api_key,
      base_url=base_url,
      default_headers=default_headers,
    )
    self.model_name = model_name
    self.max_tokens = max_tokens
    self.temperature = temperature

  def generate(self, prompt: str, **kwargs: Any) -> LLMResult:
    """调用 OpenRouter (OpenAI 兼容接口) 进行生成。"""

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

