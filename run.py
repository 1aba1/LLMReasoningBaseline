"""项目入口：对一个范式跑一次实验

后续扩展：
- 新增 CoT 方法（在 src/methods 下增加 cot.py，继承 SolverBase）
- 新增其它模型适配器（在 src/llms 下增加对应客户端，实现 BaseLLM）

"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from src.llms.openai_client import OpenAIClient
from src.methods.direct import DirectSolver
from src.utils.data_loader import load_json_dataset
from src.utils.evaluator import compute_accuracy, aggregate_token_usage


# 获取当前文件所在文件夹绝对路径
ROOT_DIR = Path(__file__).resolve().parent


def load_yaml(path: str) -> Dict[str, Any]:
  """读取 YAML 配置文件,并返回字典格式的配置数据"""
  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def read_text(path: str) -> str:
  """读取纯文本文件（如 prompt 模板）。"""
  with open(path, "r", encoding="utf-8") as f:
    return f.read()


def resolve_path(relative: str) -> str:
  """把 run_config.yaml 中的相对路径转成绝对路径。
  这样无论你从哪个工作目录执行 `python run.py`，路径都能正确解析。
  """
  return str((ROOT_DIR / relative).resolve())


def main() -> None:
  # 1. 加载环境变量
  load_dotenv()

  # 2. 读取运行总配置
  run_cfg_path = ROOT_DIR / "configs" / "run_config.yaml"
  run_cfg = load_yaml(str(run_cfg_path))

  # 获取模型与推理范式的配置
  model_cfg = load_yaml(resolve_path(run_cfg["model_config"]))
  method_cfg = load_yaml(resolve_path(run_cfg["method_config"]))

  # 获取数据的路径
  input_file = resolve_path(run_cfg["input_file"])
  prompt_file = resolve_path(run_cfg["prompt_file"])
  output_dir = resolve_path(run_cfg["output_dir"])
  max_samples = run_cfg.get("max_samples")

  # 3. 初始化模型（这里只实现 openai 一种类型）
  if model_cfg.get("model_type") != "openai":
    raise NotImplementedError("当前示例仅实现 model_type=openai 的情况。")

  # 创建openai客户端
  llm = OpenAIClient(
    model_name=model_cfg.get("model_name"),
    base_url=model_cfg.get("base_url"),
    max_tokens=model_cfg.get("max_tokens"),
    temperature=model_cfg.get("temperature"),
  )

  # 4. 初始化方法（目前只实现 direct）
  method_name = method_cfg.get("method_name")
  if method_name != "direct":
    raise NotImplementedError("当前示例仅实现 method_name=direct 的情况。")

  prompt_template = read_text(prompt_file)
  solver = DirectSolver(llm=llm, prompt_template=prompt_template)

  # 5. 加载数据集
  dataset = load_json_dataset(input_file, max_samples=max_samples)

  # 6. 为本次运行创建输出目录
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  model_name = model_cfg.get("model_name")
  run_name = run_cfg.get("run_name", "run")

  run_folder_name = f"{timestamp}_{run_name}_{method_name}_{model_name}"
  run_folder = Path(output_dir) / run_folder_name
  run_folder.mkdir(parents=True, exist_ok=True)

  full_log_path = run_folder / "full_log.jsonl"
  metrics_path = run_folder / "metrics.json"

  records = []

  # 7. 主循环：对每个样本调用一次 LLM
  print(f"开始运行实验：{run_name}")
  print(f"样本数量：{len(dataset)}")

  with open(full_log_path, "w", encoding="utf-8") as log_f:
    for i, sample in enumerate(dataset, start=1):
      print(f"正在处理样本 {i}/{len(dataset)} (id={sample.get('id')}) ...")
      result = solver.run_sample(sample)
      records.append(result)

      # 一条记录一行 JSON，方便后续增量分析
      log_f.write(json.dumps(result, ensure_ascii=False) + "\n")

  # 8. 计算整体指标
  accuracy = compute_accuracy(records)
  token_usage = aggregate_token_usage(records)

  metrics = {
    "run_name": run_name,
    "method": method_name,
    "model": model_name,
    "num_samples": len(records),
    "accuracy": accuracy,
    "token_usage": token_usage,
  }

  with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

  print("实验结束！\n")
  print("指标如下：")
  print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
