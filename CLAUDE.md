# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

τ-bench is a benchmark for evaluating Tool-Agent-User interaction in real-world domains. It simulates dynamic conversations between a user (simulated by language models) and a language agent provided with domain-specific API tools and policy guidelines. The benchmark supports two main environments: `airline` and `retail`.

## Installation and Setup

Install the package from source:
```bash
pip install -e .
```

For LangGraph agents, install with extra dependencies:
```bash
pip install -e ".[langgraph]"
```

Set up API keys as environment variables:
```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export MISTRAL_API_KEY=...
```

## Core Commands

### Running Benchmarks

Basic benchmark run:
```bash
python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --user-model gpt-4o --user-model-provider openai --user-strategy llm --max-concurrency 10
```

Run specific tasks:
```bash
python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --user-model gpt-4o --user-model-provider openai --user-strategy llm --max-concurrency 10 --task-ids 2 4 6
```

### LangGraph Agent with Tracing

Run with LangGraph agent (includes OpenTelemetry tracing):
```bash
python run.py --agent-strategy langgraph --env retail --model gpt-4o --model-provider openai --user-model gpt-4o --user-model-provider openai --user-strategy llm --max-concurrency 10
```

### Auto Error Identification

Analyze failed results:
```bash
python auto_error_identification.py --env airline --platform openai --results-path <results_file_path> --max-concurrency 16 --output-path test-auto-error-identification --max-num-failed-results 10
```

### Trace Management

Manage and analyze trace files (when using LangGraph agent):
```bash
# List all trace files
python trace_utils.py list

# Analyze a specific trace file
python trace_utils.py analyze traces/trace_file.json

# Merge multiple trace files
python trace_utils.py merge traces/trace_*.json -o merged_traces.json
```

## Architecture

### Core Components

1. **Environments (`tau_bench/envs/`)**:
   - `base.py`: Core `Env` class that handles task execution, tool management, and user interaction
   - `airline/`: Airline domain with customer service scenarios
   - `retail/`: Retail domain with e-commerce scenarios
   - Each environment contains: `env.py`, `tools/`, `data/`, `tasks*.py`, `wiki.md`

2. **Agents (`tau_bench/agents/`)**:
   - `base.py`: Abstract `Agent` class with `solve()` method
   - `tool_calling_agent.py`: Function-calling strategy agent
   - `chat_react_agent.py`: ReAct strategy agent
   - `few_shot_agent.py`: Few-shot prompting agent
   - `langgraph_tool_call_agent.py`: LangGraph-based agent with tracing
   - `memory_agent.py`: Agent with memory capabilities

3. **Model Utilities (`tau_bench/model_utils/`)**:
   - Abstraction layer for different LLM providers (OpenAI, Anthropic, Google, Mistral)
   - Uses `litellm` for unified model access

4. **User Simulation (`tau_bench/envs/user.py`)**:
   - Strategies: `llm`, `react`, `verify`, `reflection`
   - Simulates realistic user responses and behavior

### Key Types (`tau_bench/types.py`)

- `Task`: User instruction with expected actions and outputs
- `Action`: Tool call with name and parameters
- `SolveResult`: Agent's final result with reward and conversation history
- `EnvRunResult`: Complete evaluation result for a task

### Agent Strategies

- `tool-calling`: Native function-calling (recommended)
- `act`: Action-only approach
- `react`: Reasoning + Action cycles
- `few-shot`: Few-shot prompting
- `langgraph`: LangGraph-based with observability
- `memory`: Agent with memory capabilities

### User Strategies

- `llm`: Standard LLM simulation
- `react`: ReAct-style user responses
- `verify`: LLM verification step
- `reflection`: Self-reflection mechanism

## Development Guidelines

### Adding New Environments

1. Create directory in `tau_bench/envs/`
2. Implement `env.py` with environment-specific logic
3. Add tools in `tools/` directory
4. Create `tasks*.py` with task definitions
5. Write `wiki.md` with domain knowledge
6. Register in `tau_bench/envs/__init__.py`

### Adding New Agents

1. Inherit from `tau_bench.agents.base.Agent`
2. Implement `solve()` method
3. Handle tool calling and conversation flow
4. Return `SolveResult` with proper metrics

### Testing

No formal test framework is configured. Run individual components:
```bash
python tau_bench/run.py --help
python auto_error_identification.py --help
```

### Data Locations

- Environment data: `tau_bench/envs/{env}/data/`
- Historical trajectories: `./historical_trajectories/`
- Results: Configurable via `--log-dir` (default: `./results/`)

## Configuration

All configuration is handled via command-line arguments to `run.py`. Key parameters:
- `--env`: Environment (`retail`, `airline`)
- `--agent-strategy`: Agent type
- `--model`/`--model-provider`: Agent model
- `--user-model`/`--user-model-provider`: User simulator model
- `--task-split`: Data split (`train`, `test`, `dev`)
- `--max-concurrency`: Parallel execution limit

## LangGraph Agent and Tracing

The LangGraph agent provides enhanced observability through OpenTelemetry tracing. See `TRACING_SETUP.md` for detailed setup instructions.

### Quick Setup

1. Install with tracing dependencies:
```bash
pip install -e ".[langgraph]"
```

2. Start Phoenix server (optional, for visualization):
```bash
# Phoenix server should run on port 6006
```

3. Use the LangGraph agent:
```bash
python run.py --agent-strategy langgraph --env retail --model gpt-4o --model-provider openai
```

### Tracing Features

- **Phoenix Integration**: Real-time trace visualization
- **File Export**: Save traces for offline analysis
- **Multiple Formats**: JSON and OTLP export formats
- **Detailed Instrumentation**: LLM calls, tool executions, user interactions
- **Cost Tracking**: Token usage and API costs per span