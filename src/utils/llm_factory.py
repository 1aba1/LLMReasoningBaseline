"""LLM 客户端工厂：根据模型配置创建对应的 LLM 客户端。

设计目标：
- 通过 `model_type` 和 `model_config` YAML 自由切换不同模型/后端
- 不破坏现有 OpenAIClient 用法
- 后续扩展（本地/服务器、其他 OpenAI 兼容服务）只需在这里增加分支
"""
import os
from typing import Any, Dict

from src.core.llm_base import BaseLLM
from src.llms.openai_client import OpenAIClient
from src.llms.openrouter_client import OpenRouterClient


def choose_llm(model_cfg: Dict[str, Any]) -> BaseLLM:
  """根据模型配置创建对应的 LLM 客户端。

  参数
  ------
  model_cfg: Dict[str, Any]
    模型配置字典，包含 model_type、model_name、base_url 等字段

  返回
  ------
  BaseLLM
    对应的 LLM 客户端实例

  支持的 model_type:
  - openai: OpenAI 兼容的 API
  - openrouter: OpenRouter 服务
  """

  model_type = model_cfg.get("model_type")
  model_name = model_cfg.get("model_name")
  base_url = model_cfg.get("base_url")
  max_tokens = model_cfg.get("max_tokens")
  temperature = model_cfg.get("temperature")

  # 允许在配置里指定使用哪个环境变量读取 key，默认按类型给一个合理值
  api_key_env = model_cfg.get("api_key_env")
  api_key = os.getenv(api_key_env)

  if model_type == "openai":
    return OpenAIClient(
      model_name=model_name,
      base_url=base_url,
      max_tokens=max_tokens,
      temperature=temperature,
      api_key=api_key,
    )

  if model_type == "openrouter":
    # 可选：允许在 YAML 中额外传一些 OpenRouter header
    extra_headers = model_cfg.get("extra_headers") or None
    return OpenRouterClient(
      model_name=model_name,
      base_url=base_url,
      max_tokens=max_tokens,
      temperature=temperature,
      api_key=api_key,
      extra_headers=extra_headers,
    )

  # 未来如果要支持本地 Llama / 服务器模型，可以在这里继续加分支
  raise ValueError(f"不支持的 model_type: {model_type}")
