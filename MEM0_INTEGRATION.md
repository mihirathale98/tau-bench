# Mem0 Integration for Tau-Bench Memory Experiments

This document describes the mem0-based memory system integrated into tau-bench for running memory-augmented agent experiments.

## Overview

The mem0 integration enables agents to learn from past successful trajectories and use those learnings to improve performance on future tasks. This is a cleaner alternative to manual Qdrant management.

### Key Features

✅ **Policy-Aware Memory Generation** - Memories respect fixed policy guidelines from wiki.md
✅ **Quality Filtering** - Only stores high-quality trajectories (reward >= 0.8)
✅ **Comprehensive Logging** - All generated memories logged to `memory_generation_mem0.log`
✅ **Automatic Deduplication** - mem0 handles vector DB management internally
✅ **Easy Comparison** - Clean separation from original Qdrant implementation

## Architecture

### New Files Created

1. **[tau_bench/agents/mem0_module.py](tau_bench/agents/mem0_module.py)**
   - Wrapper around mem0's Memory API
   - Provides same interface as original MemModule
   - Handles embedding, storage, and retrieval

2. **[tau_bench/agents/memory_agent_mem0.py](tau_bench/agents/memory_agent_mem0.py)**
   - Memory-augmented agent using mem0
   - Generates procedural memory cards from trajectories
   - Retrieves and injects relevant memories during test time
   - **NEW**: Actually logs all generated memories
   - **NEW**: Filters by reward threshold (>= 0.8)

### Modified Files

1. **[setup.py](setup.py)** - Added `mem0ai>=0.0.1` dependency
2. **[run.py](run.py)** - Added `memory-mem0` to CLI choices
3. **[tau_bench/run.py](tau_bench/run.py)** - Added memory-mem0 agent factory and initialization

## Installation

```bash
# Install tau-bench with mem0 support
pip install -e .

# Or using uv
uv pip install -e .
```

## Usage

### Experiment 1: Baseline (No Memory)

Run tasks without any memory augmentation to establish baseline performance:

```bash
python run.py \
  --agent-strategy tool-calling \
  --env retail \
  --model gpt-4o \
  --model-provider openai \
  --user-model gpt-4o \
  --user-model-provider openai \
  --task-split test \
  --max-concurrency 10 \
  --log-dir results/baseline_no_memory
```

### Experiment 2: Train Memory Bank

Run on **training split** to populate mem0 with procedural memories:

```bash
python run.py \
  --agent-strategy memory-mem0 \
  --env retail \
  --model gpt-4o \
  --model-provider openai \
  --user-model gpt-4o \
  --user-model-provider openai \
  --task-split train \
  --max-concurrency 10 \
  --log-dir results/train_mem0_memory_bank
```

**What happens:**
- Agent solves training tasks
- After each task, generates a procedural memory card
- Only stores memories with reward >= 0.8 (configurable in code)
- All memories logged to `memory_generation_mem0.log`
- Memories stored in mem0 vector DB with metadata

### Experiment 3: Test with Memory Retrieval

Run on **test split** using memories created during training:

```bash
python run.py \
  --agent-strategy memory-mem0 \
  --env retail \
  --model gpt-4o \
  --model-provider openai \
  --user-model gpt-4o \
  --user-model-provider openai \
  --task-split test \
  --max-concurrency 10 \
  --log-dir results/test_mem0_with_memory
```

**What happens:**
- Agent retrieves top 4 relevant memories for each task
- Memories injected into system prompt as "past experiences"
- Agent uses memories to guide decision-making
- Performance compared against baseline

## Memory Format

Memories are generated using GPT-4o with the following structure:

```
**User Intent**: [One-line generalized intent]

**Steps-tool mapping**:
- Step 1: [Tool used] for [action]
- Step 2: [Tool used] for [action]
...

**Learning**: [1-2 lines of key takeaways]

**Correction needed**: [1-2 lines of what could be improved]

**Deviation**: [Any violations of fixed policies, or "None"]
```

### Example Memory

```
**User Intent**: Exchange delivered items for different product variants

**Steps-tool mapping**:
- Step 1: find_user_id_by_name_zip for authentication
- Step 2: get_order_details to verify order status is "delivered"
- Step 3: get_product_details to find available variants
- Step 4: exchange_delivered_order_items with user confirmation

**Learning**: Always verify order status before attempting exchange.
Confirm all items with user before making the exchange call.

**Correction needed**: Should explicitly confirm payment method for
price difference handling.

**Deviation**: None
```

## Configuration

### Memory Quality Threshold

Edit [tau_bench/agents/memory_agent_mem0.py](tau_bench/agents/memory_agent_mem0.py):

```python
MIN_REWARD_THRESHOLD = 0.8  # Only store/retrieve memories with reward >= 0.8
```

### Retrieval Settings

```python
MEMORY_RETRIEVAL_LIMIT = 4  # Number of memories to retrieve per task
```

### Mem0 Backend

Edit [tau_bench/agents/mem0_module.py](tau_bench/agents/mem0_module.py) to configure vector store:

```python
config = {
    "vector_store": {
        "provider": "qdrant",  # or "pinecone", "chroma", etc.
        "config": {
            "collection_name": collection_name,
            "embedding_model_dims": 1536,
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    }
}
```

## Analyzing Results

### View Generated Memories

```bash
# View all generated memories
cat memory_generation_mem0.log

# View only stored memories (reward >= 0.8)
grep "✓ Stored" memory_generation_mem0.log

# View only skipped memories (reward < 0.8)
grep "✗ Skipped" memory_generation_mem0.log

# View memory content for perfect trajectories
grep -A 20 "Reward: 1.00" memory_generation_mem0.log
```

### Compare Performance

```bash
# Baseline (no memory)
results/baseline_no_memory/*_pass_hat_ks.json

# With mem0 memory
results/test_mem0_with_memory/*_pass_hat_ks.json
```

Look for improvements in Pass@k metrics.

## How Memory Improves Performance

### Policy Adherence

Memories explicitly include the fixed policy guidelines (wiki.md), ensuring:
- Learnings don't contradict policies
- Deviations are flagged in the "Deviation" section
- Only policy-compliant memories (reward >= 0.8) are stored

### Procedural Learning

Memories capture:
1. **Tool sequencing**: Which tools to use in what order
2. **Verification steps**: Check order status, confirm details, etc.
3. **User confirmation**: When to ask for explicit user approval
4. **Error patterns**: What NOT to do based on failed attempts

### Generalization

Memory generation prompt ensures:
- No PII or specific IDs in memories (generalizable patterns)
- Concise format (<250 tokens)
- Reusable across similar scenarios

## Troubleshooting

### mem0 Not Installed

```bash
pip install mem0ai
# or
uv pip install mem0ai
```

### No Memories Retrieved

Check:
1. Did you run training first? (`--task-split train`)
2. Were any memories stored? Check `memory_generation_mem0.log`
3. Is reward threshold too high? Lower `MIN_REWARD_THRESHOLD`
4. Is intent similarity too strict? Increase `MEMORY_RETRIEVAL_LIMIT`

### Poor Performance with Memory

Possible causes:
1. Training set too small (not enough diverse memories)
2. Training performance too low (few memories with reward >= 0.8)
3. Memory prompt needs tuning (improve `generate_traj_summary()`)
4. Retrieved memories not relevant (adjust intent summarization)

## Next Steps

### Phase 1: Baseline Comparison (Current)
- ✅ Implement mem0 integration
- ✅ Add logging and filtering
- 🔄 Run experiments 1-3
- 🔄 Compare baseline vs mem0 performance

### Phase 2: Prompt Engineering
- Tune memory generation prompt for better quality
- Experiment with JSON vs plain text format
- Add examples of good/bad memories
- Test different temperatures (0.0 vs 0.6)

### Phase 3: Advanced Features
- Add policy validation layer
- Implement memory quality scoring
- Multi-hop reasoning over memories
- Fine-tune retrieval (reranking, hybrid search)

## Comparison with Original Implementation

| Feature | Original (Qdrant) | New (mem0) |
|---------|------------------|------------|
| Vector DB | Manual Qdrant setup | Managed by mem0 |
| Logging | ❌ Not implemented | ✅ Comprehensive logs |
| Quality Filter | ❌ Stores all (reward=0) | ✅ Only reward >= 0.8 |
| Dependencies | qdrant-client, openai | mem0ai (includes both) |
| Deduplication | Manual | Automatic |
| Retrieval Filter | Optional (unused) | Active (reward threshold) |
| Policy Awareness | ✅ Included | ✅ Included (preserved) |

## Contact

For questions or issues with the mem0 integration, check:
- Memory logs: `memory_generation_mem0.log`
- mem0 docs: https://github.com/mem0ai/mem0
- tau-bench docs: [CLAUDE.md](CLAUDE.md)
