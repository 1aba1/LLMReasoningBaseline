## 项目简介

这是一个用于对比不同大模型推理范式（baseline）的实验框架。当前版本先实现：

- **方法**：LLM Direct Output（直接输出最终答案）
- **模型**：OpenAI / OpenAI 兼容 API（例如一些云厂商或实验室自建的兼容服务）

后续你可以在此基础上逐步扩展：

- 新增 **CoT（Chain-of-Thought）**、ReAct 等推理范式
- 新增本地部署/开源模型的适配器

---

## 目录结构与文件说明

当前项目根目录下的主要内容如下（以你本地实际结构为准）：

- **.env**: 环境变量配置文件（API Key 等）
- **configs/**: 各种 YAML 配置
  - **models/**
    - `openai_gpt4o.yaml`: 模型配置文件（被 `run.py` 通过 `configs/run_config.yaml` 间接加载）
  - **methods/**
    - `direct.yaml`: 方法（推理范式）配置文件（同样由 `run.py` 通过 `configs/run_config.yaml` 加载）
  - `run_config.yaml`: 运行总配置，决定本次实验用哪种模型、哪种方法、哪个数据集、输出到哪里
- **data/**
  - **inputs/**
    - `math_test_sample.json`: 示例数据集
  - **prompts/**
    - `direct_prompt.txt`: Direct 方法使用的提示词模板
  - **outputs/**
    - 每次运行会在此目录下新建一个子目录，例如：`20251216_132850_demo_direct_run_direct_gpt-4o-mini/`
      - `full_log.jsonl`: 每个样本的完整记录（Prompt、模型原始输出、解析后的答案、token 使用等）
      - `metrics.json`（当前示例运行输出可能只包含 `full_log.jsonl`，但推荐后续保留该文件用于汇总指标）
- **src/**: 核心代码
  - **core/**: 抽象基类
    - `llm_base.py`: LLM 抽象基类 `BaseLLM`，统一大模型调用接口
    - `solver_base.py`: 推理范式抽象基类 `SolverBase`
  - **llms/**: 具体模型适配器
    - `openai_client.py`: OpenAI / OpenAI 兼容 API 适配，实现 `BaseLLM` 的 `generate` 方法
  - **methods/**: 推理范式实现
    - `direct.py`: Direct 输出方法，实现 `SolverBase`，负责构造 prompt 和解析答案
  - **utils/**: 工具库
    - `answer_extractor.py`: 从模型输出中提取 A/B/C/D 答案
    - `token_counter.py`: 汇总 token 使用情况
    - `evaluator.py`: 计算准确率和汇总 token
    - `data_loader.py`: 加载 JSON 数据集
- **run.py**: 项目入口脚本（主运行逻辑）
- **requirements.txt**: Python 依赖列表（`openai`、`python-dotenv`、`PyYAML` 等）

> 说明：之前 README 中提到的根目录 `logs/`、`outputs/` 目录，本示例中并未真正创建；
> 实际输出目录为 `data/outputs/`，且目前日志主要体现在 `data/outputs/.../full_log.jsonl` 中。

---

## 文件关系与运行链路

这一节专门梳理“从入口到模型调用”的完整链条，方便你理解每个文件的角色，以及 `openai_gpt4o.yaml` 等配置是如何被实际使用的。

### 1. 顶层入口：`run.py`

- **职责**：
  - 读取 `.env` 和各类 YAML 配置
  - 初始化模型客户端 (`OpenAIClient`)
  - 初始化推理方法 (`DirectSolver`)
  - 加载数据集、循环调用模型、保存结果和指标
- **关键逻辑（简化描述）**：
  - 从 `configs/run_config.yaml` 中读取：
    - `model_config`: 模型配置文件路径（例如 `configs/models/openai_gpt4o.yaml`）
    - `method_config`: 方法配置文件路径（例如 `configs/methods/direct.yaml`）
    - `input_file`: 数据集路径（例如 `data/inputs/math_test_sample.json`）
    - `prompt_file`: 提示词模板路径（例如 `data/prompts/direct_prompt.txt`）
    - `output_dir`: 输出目录（当前为 `data/outputs`）
  - 使用 `load_yaml` 读取上面两个 YAML：
    - 得到 `model_cfg`（来自 `openai_gpt4o.yaml`）
    - 得到 `method_cfg`（来自 `direct.yaml`）
  - 根据 `model_cfg` 初始化 `OpenAIClient`：
    - `model_name`、`base_url`、`max_tokens`、`temperature` 等参数
  - 根据 `method_cfg` 确定使用的方法名称（当前仅支持 `direct`），然后：
    - 读取 `prompt_file` 得到 `prompt_template`
    - 创建 `DirectSolver(llm=llm, prompt_template=prompt_template)`
  - 使用 `load_json_dataset` 加载数据集
  - 遍历每个样本，调用 `solver.run_sample(sample)`，将结果写入 `full_log.jsonl`
  - 使用 `compute_accuracy` 和 `aggregate_token_usage` 计算整体指标，写入 `metrics.json`

**因此：`openai_gpt4o.yaml` 并不是“孤立存在”的文件，而是由 `configs/run_config.yaml → run.py` 这条链路在运行时真正加载和使用。**

### 2. 配置文件链路

- **`configs/run_config.yaml`**
  - 决定“本次实验用哪些子配置”：
    - 使用哪个模型配置文件：`model_config: "configs/models/openai_gpt4o.yaml"`
    - 使用哪种推理方法配置：`method_config: "configs/methods/direct.yaml"`
    - 使用哪个数据集、哪个 prompt 模板、输出到哪个目录
- **`configs/models/openai_gpt4o.yaml`**
  - 提供模型相关参数：
    - `model_type: openai`：当前代码只实现了这一种类型
    - `model_name: gpt-4o-mini`：传给 OpenAI SDK 的 `model` 字段
    - `base_url: ${OPENAI_BASE_URL}`：从环境变量中读取兼容 OpenAI 的服务地址
    - `max_tokens`、`temperature`：作为默认生成参数传入 `OpenAIClient`
- **`configs/methods/direct.yaml`**
  - 提供方法层面的元信息：
    - `method_name: direct`：告诉 `run.py` 需要实例化 `DirectSolver`
    - 其它参数（如 `max_output_tokens`、`temperature`）可以在后续扩展中传递给 Solver 或 LLM

### 3. 模型适配层：`src/llms/openai_client.py`

- **`OpenAIClient`** 继承自 `BaseLLM`：
  - 在构造函数中：
    - 使用 `load_dotenv()` 加载 `.env`
    - 从环境变量读取 `OPENAI_API_KEY` 和可选的 `OPENAI_BASE_URL`
    - 初始化 `OpenAI` 客户端对象
  - 在 `generate` 方法中：
    - 接收上层传入的 `prompt`
    - 调用 `client.chat.completions.create(...)` 完成一次对话生成
    - 封装为 `LLMResult` 返回给上层

### 4. 方法层：`src/methods/direct.py`

- **`DirectSolver(SolverBase)`**：
  - `build_prompt`：
    - 使用 `prompt_template.format(question=question)` 将样本中的 `question` 字段填入模板
  - `parse_answer`：
    - 调用 `extract_choice_answer(llm_result.text)` 从模型输出文本中提取 A/B/C/D
  - `run_sample` 由基类 `SolverBase` 提供：
    - 负责串联 `build_prompt → llm.generate → parse_answer` 并组织返回字典

### 5. 工具层：`src/utils/*`

- **`data_loader.py`**：
  - `load_json_dataset(path, max_samples)`：读取列表形式的 JSON 数据集
- **`answer_extractor.py`**：
  - `extract_choice_answer(text)`：从模型输出中用规则/正则提取单个选项字母
- **`evaluator.py`**：
  - `compute_accuracy(records)`：根据 `gold_answer` 与 `pred_answer` 计算准确率
  - `aggregate_token_usage(records)`：调用 `token_counter.sum_tokens` 汇总 token 使用
- **`token_counter.py`**：
  - `sum_tokens(usages)`：对每条记录中的 `prompt_tokens`、`completion_tokens`、`total_tokens` 做求和

---

## 环境准备

### 1. 创建并激活虚拟环境（推荐）

以 Windows + PowerShell 为例：

```bash
cd "d:/Project/LearingAboutPython/大模型推理范式baseline/测试direct+CoT"

# 创建虚拟环境（可自定义 env 名字）
python -m venv .venv

# 激活虚拟环境
.venv\Scripts\activate
```

如果你使用的是 conda，也可以：

```bash
conda create -n llm-baseline python=3.10 -y
conda activate llm-baseline
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

---

## 配置环境变量（.env）

项目根目录已经给出一个 `.env` 示例：

```bash
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4o-mini
```

请将 `YOUR_OPENAI_API_KEY_HERE` 替换为你自己的 API Key：

- 如果你使用 **官方 OpenAI**，保持 `OPENAI_BASE_URL` 为 `https://api.openai.com/v1` 即可。
- 如果你使用 **兼容 OpenAI 的第三方服务/实验室自建服务**，把 `OPENAI_BASE_URL` 改成对应的地址，例如：
  - `https://api.deepseek.com/v1`
  - `https://your-lab-server/v1`

模型名称（`OPENAI_MODEL_NAME`）需要和服务端支持的 model 对齐，例如：

- `gpt-4.1`
- `gpt-4o`
- 某些兼容服务的自定义名字，如 `deepseek-chat` 等

> 小贴士：`.env` 文件不会被代码直接提交到远程仓库（如果你后续初始化 git 的话），用于安全存放秘钥信息。

---

## 配置运行参数（YAML）

### 1. 模型配置：`configs/models/openai_gpt4o.yaml`

关键字段：

- **model_type**: 目前固定为 `openai`
- **model_name**: 模型名称
- **base_url**: 从环境变量读取 `${OPENAI_BASE_URL}`
- **max_tokens/temperature**: 默认生成参数

你可以根据自己的情况修改：

```yaml
model_type: openai
model_name: gpt-4o-mini
base_url: ${OPENAI_BASE_URL}
max_tokens: 512
temperature: 0.0
```

### 2. 方法配置：`configs/methods/direct.yaml`

目前只需要关心：

- **method_name**: `direct`
- 其它参数（如 `max_output_tokens`、`temperature`）预留给后续扩展；
  当前示例中主要通过模型配置控制 `max_tokens/temperature`。

```yaml
method_name: direct
max_output_tokens: 256
temperature: 0.0
```

### 3. 运行总配置：`configs/run_config.yaml`

主要字段：

- **run_name**: 本次实验的名字
- **model_config**: 指向上面的模型配置文件
- **method_config**: 指向方法配置文件
- **input_file**: 数据集路径
- **prompt_file**: 提示词模板路径
- **output_dir**: 输出目录（当前为 `data/outputs`）
- **max_samples**: 为 `null` 表示全部样本；设为一个数字可以只跑前 N 条用于快速测试

```yaml
run_name: "demo_direct_run"

model_config: "configs/models/openai_gpt4o.yaml"
method_config: "configs/methods/direct.yaml"

input_file: "data/inputs/math_test_sample.json"
prompt_file: "data/prompts/direct_prompt.txt"

output_dir: "data/outputs"
max_samples: null
```

---

## 数据格式与提示词

### 1. 数据集示例：`data/inputs/math_test_sample.json`

目前的示例数据是一个简单的数学选择题列表：

```json
[
  {
    "id": 1,
    "question": "Let k be the number of real solutions of the equation e^x + x - 2 = 0 in the interval [0, 1], and let n be the number of real solutions that are not in [0, 1]. Which of the following is true?\nA).k = 0 and n = 1   B).k = 1 and n = 0   C).k = n = 1   D).k > 1",
    "answer": "B"
  },
  {
    "id": 2,
    "question": "If 2x + 3 = 11, what is the value of x?\nA). 3   B). 4   C). 5   D). 6",
    "answer": "B"
  }
]
```

如果你有自己的数据集，只要保持同样的字段结构即可：

- **id**: 唯一编号（整数或字符串均可）
- **question**: 题目文本（可以包含换行和选项）
- **answer**: 标准答案（A/B/C/D）

### 2. 提示词模板：`data/prompts/direct_prompt.txt`

示例内容：

```text
你是一名擅长数学选择题的助教，请直接给出最终的选项答案（A/B/C/D），不要输出多余解释。

题目如下：

{question}

请仅回答一个大写字母选项（A/B/C/D）。
```

其中 `{question}` 是占位符，代码在运行时会将样本中的 `question` 字段填充进去。

---

## 如何运行一次完整实验

1. **确保已激活虚拟环境，并安装依赖**：

   ```bash
   cd "d:/Project/LearingAboutPython/大模型推理范式baseline/测试direct+CoT"
   .venv\Scripts\activate      # 或 conda activate llm-baseline
   pip install -r requirements.txt
   ```

2. **配置好 `.env` 中的 `OPENAI_API_KEY`**，并根据需要修改 `OPENAI_BASE_URL` 和 `OPENAI_MODEL_NAME`。

3. **根据需要调整 `configs/run_config.yaml`**（比如换数据集、改输出目录）。

4. **运行入口脚本**：

   ```bash
   python run.py
   ```

5. 运行成功后，你会在 `data/outputs/` 目录下看到一个新建的子文件夹，例如：

   ```text
   data/outputs/
     20251216_132850_demo_direct_run_direct_gpt-4o-mini/
       full_log.jsonl
       metrics.json    # 如果你后续保留了指标输出逻辑
   ```

   - `full_log.jsonl`: 每行一个 JSON，记录了每个样本的：
     - `id`
     - `question`
     - `gold_answer`（标准答案）
     - `pred_answer`（模型预测）
     - `raw_output`（模型原始输出文本）
     - `usage`（token 使用情况）
   - `metrics.json`: 整体评估指标，例如：

     ```json
     {
       "run_name": "demo_direct_run",
       "method": "direct",
       "model": "gpt-4o-mini",
       "num_samples": 2,
       "accuracy": 1.0,
       "token_usage": {
         "prompt_tokens": 100,
         "completion_tokens": 50,
         "total_tokens": 150
       }
     }
     ```

---

## 下一步可以怎么扩展？

1. **实现 CoT（single-agent reasoning augmentation）方法**：

   - 在 `data/prompts/` 中新增 `cot_prompt.txt`，让模型显式输出推理过程和最终答案。
   - 在 `src/methods/` 中新增 `cot.py`，继承 `SolverBase`：
     - `build_prompt`：构造带有“请先一步步推理再给出答案”的 prompt
     - `parse_answer`：先保留完整推理过程到 `raw_output`，再用正则提取最终的 A/B/C/D
   - 在 `configs/methods/` 中新增 `cot.yaml`，并在 `run_config.yaml` 中切换 `method_config`。

2. **接入本地/开源模型**：

   - 在 `src/llms/` 下新增比如 `local_client.py`，实现 `BaseLLM`：
     - 在 `generate` 里通过 HTTP、gRPC 或本地推理引擎调用模型
   - 在 `configs/models/` 中增加新的 YAML（比如 `local_llama.yaml`），并在 `run_config.yaml` 中切换。

3. **增加日志与监控**：

   - 使用 Python `logging` 模块，把每次调用信息输出到一个单独的 `logs/` 目录
   - 记录失败请求、重试等信息，便于排查问题。

如果你愿意，下一步我可以帮你：

- 补全 CoT 方法（包括 prompt 和解析逻辑）
- 或者帮你接入你们实验室服务器上的本地大模型服务。
