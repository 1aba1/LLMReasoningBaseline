#测试每一种推理范式，在math_test_sample数据集上运行，使用openai/gpt-4.1-mini模型，每种方法仅测试2个样本。
python main.py \
methods=Direct \
filename="math_test_sample" \
models=openrouter \
models.model_name="openai/gpt-4.1-mini" \
max_samples=2

python main.py \
methods=CoT \
filename="math_test_sample" \
models=openrouter \
models.model_name="openai/gpt-4.1-mini" \
max_samples=2

python main.py \
methods=React \
filename="math_test_sample" \
models=openrouter \
models.model_name="openai/gpt-4.1-mini" \
max_samples=2

python main.py \
methods=Debate \
filename="math_test_sample" \
models=openrouter \
models.model_name="openai/gpt-4.1-mini" \
max_samples=2