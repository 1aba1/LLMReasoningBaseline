#一次实验的运行脚本。每次实验可以选择以下参数
#
#methods（推理范式）：Direct、CoT、React、Debate
#filename(测试数据集）：math_test_sample、mmlu_college_mathematics
#models(大模型提供商)：openai、openrouter
#models.model_name（调用的模型，需要根据提供商来确定模型命名）：gpt-4o-mini、deepseek/deepseek-v3.2
#max_samples（输入样例数目）：为空则代表运行全部案例
#(可选参数)run_name：为实验起名
#
#实验后可以查看的信息：
#主要配置信息和实验指标结果在data/outputs里的metrics.jsonl查看、
# 模型原始交互信息在data/outputs里的full_log.jsonl里查看、
# 配置的完整信息可以在data/hydra_outputs查看

python main.py \
methods=direct \
filename="math_test_sample" \
models=openrouter \
models.model_name="deepseek/deepseek-v3.2" \
