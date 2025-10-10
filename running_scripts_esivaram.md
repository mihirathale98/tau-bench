```bash
uv run python run.py \
  --agent-strategy tool-calling \
  --env retail \
  --model gpt-4.1-mini \
  --model-provider openai \
  --user-model gpt-4o \
  --user-model-provider openai \
  --user-strategy llm \
  --max-concurrency 50 \
  --task-split test \
  --num-trials 4 \
  --log-dir results/baselines/4.1-mini-retail-test-num-trials-4
```


# Train on retail (500 tasks)
```bash
uv run python run.py \
  --agent-strategy memory-mem0 \
  --env retail \
  --model gpt-4.1-mini \
  --model-provider openai \
  --user-model gpt-4o \
  --user-model-provider openai \
  --task-split train \
  --max-concurrency 50 \
  --log-dir results/mem0_experiments/retail_train
```

# Test on retail (460 tasks)
```bash
uv run python run.py \
  --agent-strategy memory-mem0 \
  --env retail \
  --model gpt-4.1-mini \
  --model-provider openai \
  --user-model gpt-4o \
  --user-model-provider openai \
  --task-split test \
  --max-concurrency 50 \
  --num-trials 4 \
  --log-dir results/mem0_experiments/retail_test
```

