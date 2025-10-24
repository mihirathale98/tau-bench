#!/bin/bash

# export AGENT_BASE_URL="https://api.openai.com/v1"
export AGENT_BASE_URL="http://localhost:8108/v1"

NUM_TRIALS=2
# MODELS=("gpt-4.1" "gpt-5-nano" "gpt-4o" "gpt-4.1-mini")
# MODELS=("gpt-5-nano" "gpt-4o" "gpt-4.1-mini")
MODELS=("gpt-4.1-mini")
BUDGET=4

DIR="latency_results/openai_its"


echo "Starting benchmark runs..."
echo "Total trials: $NUM_TRIALS"
echo "Models to test: ${MODELS[*]}"
echo "================================"

for i in $(seq 2 $NUM_TRIALS); do
  # OUTPUT_FOLDER="res_retail_openai_4.1-mini-temp-1.0_num_trials_8_trial_$i"
  # OUTPUT_FOLDER="results/res_airline_openai_gpt-4.1-mini_its-judge-4.1-mini_budget_${BUDGET}_trial_$i"
  echo ""
  echo "Trial $i/$NUM_TRIALS - Output folder: $OUTPUT_FOLDER"
  
  for model in "${MODELS[@]}"; do
    OUTPUT_FOLDER="$DIR/$model-trial_$i"

    echo "  Running model: $model"
    python run.py \
      --agent-strategy tool-calling \
      --env airline \
      --model "$model" \
      --model-provider openai \
      --user-model gpt-4o \
      --user-model-provider openai \
      --user-strategy llm \
      --max-concurrency 115 \
      --temperature 1.0 \
      --num-trials 5 \
      --log-dir $OUTPUT_FOLDER \
      --task-split test \
      --max-num-steps 30 \
      --budget $BUDGET 
    echo "  Completed model: $model"
  done
  echo "Completed trial $i/$NUM_TRIALS"
done

echo ""
echo "================================"
echo "All benchmark runs completed!"