#!/bin/bash

export AGENT_BASE_URL="https://api.openai.com/v1"
# export AGENT_BASE_URL="http://localhost:8108/v1"

NUM_TRIALS=2
# MODELS=("gpt-4.1" "gpt-5-nano" "gpt-4o" "gpt-4.1-mini")
# MODELS=("gpt-5-nano" "gpt-4o" "gpt-4.1-mini")
MODELS=("gpt-4.1-mini")
BUDGET=4

echo "Starting benchmark runs..."
echo "Total trials: $NUM_TRIALS"
echo "Models to test: ${MODELS[*]}"
echo "================================"

for i in $(seq 2 $NUM_TRIALS); do
  # OUTPUT_FOLDER="res_airline_openai_its_judge_mini_4.1-mini_budget_${BUDGET}_trial_$i"
  OUTPUT_FOLDER="res_retail_openai_gpt-4.1-mini_$i"
  echo ""
  echo "Trial $i/$NUM_TRIALS - Output folder: $OUTPUT_FOLDER"
  
  for model in "${MODELS[@]}"; do
    echo "  Running model: $model"
    python run.py \
      --agent-strategy tool-calling \
      --env retail \
      --model "$model" \
      --model-provider openai \
      --user-model gpt-4o \
      --user-model-provider openai \
      --user-strategy llm \
      --max-concurrency 115 \
      --temperature 0 \
      --num-trials 5 \
      --log-dir $OUTPUT_FOLDER \
      --task-split test \
      # --budget $BUDGET \
      --max-num-steps 30 
    echo "  Completed model: $model"
  done
  echo "Completed trial $i/$NUM_TRIALS"
done

echo ""
echo "================================"
echo "All benchmark runs completed!"