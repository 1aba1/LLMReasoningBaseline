# 大模型推理范式基准测试框架

这是一个用于测试和比较不同大模型推理范式（如Direct、CoT、ReAct、Debate）的基准测试框架。

## ✨ 项目亮点

✅ **多推理范式统一评测**

- 🧠 Direct（直接推理）
- 🧩 CoT（Chain-of-Thought）
- 🤖 ReAct（Reason + Act）
- 🗣️ Debate（多智能体辩论）

✅ **多模型后端支持**（可扩展）

- 🔵 OpenAI
- 🟣 OpenRouter

✅ **科研友好的配置系统**

- ⚙️ 基于 **Hydra** 的层级化配置
- 一行命令切换 *模型 / 方法 / 数据集*

✅ **自动化评测指标**

- 📊 准确率（Accuracy）
- 💰 Token 使用量统计
- 🧾 完整 JSONL 日志，便于后处理

## 📁 项目结构

```
├── configs/             # 配置文件目录
│   ├── config.yaml      # 主配置文件
│   ├── methods/         # 推理方法配置
│   │   ├── CoT.yaml     # CoT方法配置
│   │   ├── Debate.yaml  # Debate方法配置
│   │   ├── Direct.yaml  # Direct方法配置
│   │   └── React.yaml   # React方法配置
│   └── models/          # 模型配置
│       ├── openai.yaml  # OpenAI模型配置
│       └── openrouter.yaml  # OpenRouter模型配置
├── data/                # 数据目录
│   ├── hydra_outputs/   # Hydra配置输出
│   ├── inputs/          # 输入数据集
│   ├── outputs/         # 结果输出
│   └── prompts/         # 提示词模板
├── src/                 # 源代码目录
│   ├── core/            # 核心模块
│   ├── llms/            # LLM客户端实现
│   ├── methods/         # 推理方法实现
│   └── utils/           # 工具函数
└── .env                 # (需要自行创建)存放隐私秘钥
├── main.py              # 项目入口
├── README.md            # 项目说明
├── requirements.txt     # 依赖声明
└── run.sh               # 运行脚本
```

## ⚡快速开始

### 1. 环境准备

#### 1.1安装依赖

```bash
pip install -r requirements.txt
```

#### 1.2配置环境变量

创建`.env`文件，添加API密钥：

```
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key

# OpenRouter API配置（可选）
OPENROUTER_API_KEY=your_openrouter_api_key
```

### 2. 运行方式

#### 2.1 直接运行

```bash
python main.py
```

默认使用OpenAI模型和Direct推理方法。

##### 切换推理方法

```bash
# 使用CoT方法
python main.py methods=CoT

# 使用ReAct方法
python main.py methods=React

# 使用Debate方法
python main.py methods=Debate
```

##### 切换模型

```bash
# 使用OpenRouter模型
python main.py models=openrouter
```

##### 自定义运行名称

```bash
python main.py run_name=demo_CoT methods=CoT
```

##### 限制样本数量（快速测试）

```bash
python main.py max_samples=5
```

#### 2.1 脚本运行 (Linux环境)

```
bash run.sh
```

##### **每次实验可以修改run.sh脚本或自行创建新脚本并选择以下参数**（更多信息查看config.yaml）

- models(大模型提供商)：openai、openrouter
- methods（推理范式）：Direct、CoT、React、Debate
- filename(测试数据集）：math_test_sample、mmlu_college_mathematics
- models.model_name（调用的模型，需要根据提供商来确定模型命名）：gpt-4o-mini、deepseek/deepseek-v3.2
- max_samples（输入样例数目）：为空则代表运行全部案例
- (可选参数)run_name：为实验起名

## 配置文件说明

### 主配置文件（config.yaml）

```yaml
defaults:
  - models: openai          # 默认使用OpenAI模型
  - methods: Direct         # 默认使用Direct方法
  - _self_

# 实验名称
run_name: ""

# 数据集配置
filename: "math_test_sample"
input_file: "data/inputs/${filename}.json"
prompt_file: ${methods.prompt_file}

# 输出目录
output_dir: "data/outputs"

# 最大样本数（null表示全部）
max_samples: null
```

### 推理方法配置（如CoT.yaml）

```yaml
method_name: CoT
prompt_file: "data/prompts/CoT_prompt.txt"
```

### 模型配置（如openai.yaml）

```yaml
model_type: openai
model_name: "gpt-4o-mini"
temperature: 0.7
```

## 结果分析

### 输出文件

运行完成后，结果将保存在`data/outputs/`目录下，每个运行生成一个时间戳命名的文件夹：

- `full_log.jsonl`：包含所有样本的详细结果
- `metrics.json`：汇总指标（准确率、Token使用情况等）

### 指标说明

- **准确率**：模型预测正确的样本比例
- **Token使用量**：
  - `prompt_tokens`：输入提示词的Token数量
  - `completion_tokens`：模型输出的Token数量
  - `total_tokens`：总Token数量

## 扩展框架

### 添加新的推理范式

1. 在`src/methods/`目录下创建新的Python文件（如`NewMethod.py`）
2. 继承`SolverBase`类并实现`solve`方法
3. 在`configs/methods/`目录下创建对应的配置文件（如`NewMethod.yaml`）
4. 在`data/prompts/`目录下创建对应的提示词模板（如`NewMethod_prompt.txt`）

### 添加新的模型

1. 在`src/llms/`目录下创建新的模型客户端文件
2. 继承`BaseLLM`类并实现`generate`方法
3. 在`configs/models/`目录下创建对应的配置文件
