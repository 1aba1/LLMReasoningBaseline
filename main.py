import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

from dotenv import load_dotenv
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from src.utils.config_loader import read_text
from src.utils.data_loader import load_json_dataset
from src.utils.evaluator import compute_accuracy, aggregate_token_usage
from src.utils.llm_factory import choose_llm
from src.utils.resolve_path import resolve_path
from src.utils.solver_factory import create_solver

# 获取当前文件所在文件夹绝对路径（项目根目录）
ROOT_DIR = Path(__file__).resolve().parent


# 定义配置的「结构」给Hydra用，
@dataclass
class RunConfig:
  run_name: str          # 实验名称（比如"demo_CoT"）
  input_file: str        # 数据集文件路径
  prompt_file: str       # 提示词文件路径
  output_dir: str        # 结果输出文件夹
  max_samples: Any       # 最多跑多少个样本（可选，比如只跑10个测试）
  models: Dict[str, Any] # 模型配置（比如模型类型、API地址）
  methods: Dict[str, Any]# 方法配置（比如用direct还是CoT）

# 把配置结构注册到Hydra
cs = ConfigStore.instance()
cs.store(name="run_config_schema", node=RunConfig)

#Hydra 会自动把配置文件的内容加载到cfg里，供函数使用
@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
  # 1. 加载环境变量
  load_dotenv()

  # 2. 将Hydra 的配置格式（DictConfig）转成普通 Python 字典
  model_cfg: Dict[str, Any] = OmegaConf.to_container(cfg.models, resolve=True)
  method_cfg: Dict[str, Any] = OmegaConf.to_container(cfg.methods, resolve=True)

  # 3. 解析运行相关路径（统一从项目根目录出发，避免 Hydra 改变工作目录的影响）
  input_file = resolve_path(ROOT_DIR, cfg.input_file)  # 数据集路径
  prompt_file = resolve_path(ROOT_DIR, cfg.prompt_file)  # 提示词路径
  output_dir = resolve_path(ROOT_DIR, cfg.output_dir)  # 结果输出路径
  max_samples = cfg.get("max_samples")  # 最多跑多少样本（可选）

  # 4. 初始化模型（通过工厂方法，根据 model_type 自动选择具体客户端）
  llm = choose_llm(model_cfg)

  # 5. 初始化方法（根据 method_name 选择对应 Solver）
  method_name = method_cfg.get("method_name")
  run_name = cfg.run_name
  prompt_template = read_text(prompt_file)
  solver = create_solver(method_cfg=method_cfg, llm=llm, prompt_template=prompt_template)

  # 6. 加载数据集
  dataset = load_json_dataset(input_file, max_samples=max_samples)

  # 7. 为本次运行创建输出目录
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  model_type = model_cfg.get("model_type")
  model_name = model_cfg.get("model_name")

  run_folder_name = f"{timestamp}_{run_name}_{method_name}_{model_type}_{model_name}"
  run_folder = Path(output_dir) / run_folder_name
  run_folder.mkdir(parents=True, exist_ok=True)

  full_log_path = run_folder / "full_log.jsonl"
  metrics_path = run_folder / "metrics.json"

  records = []

  # 8. 主循环：对每个样本调用一次推理范式
  print(f"开始运行实验:{run_name}")
  print(f"推理范式:{method_name} | model:{model_type}/{model_name}")
  print(f"实验样本:{len(dataset)}")

  with open(full_log_path, "w", encoding="utf-8") as log_f:
    for i, sample in enumerate(dataset, start=1):
      print(f"正在处理样本{i}/{len(dataset)} (id={sample.get('id')}) ...")
      result = solver.run_sample(sample)
      records.append(result)

      # 一条记录一行 JSON，方便后续增量分析
      log_f.write(json.dumps(result, ensure_ascii=False) + "\n")

  # 9. 计算整体指标
  accuracy = compute_accuracy(records)
  token_usage = aggregate_token_usage(records)

  metrics = {
    "run_name": run_name,
    "method": method_name,
    "data_file":cfg.input_file,
    "model supplier": model_type,
    "model_name": model_name,
    "num_samples": len(records),
    "accuracy": accuracy,
    "total_token_usage": token_usage,
  }

  with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

  print("!实验结束!\n")
  print("指标如下:")
  print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  import logging
  # 不显示httpx 的日志信息
  logging.getLogger("httpx").setLevel(logging.WARNING)
  main()
