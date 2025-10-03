# Enhanced OpenTelemetry Tracing for Tau-Bench

This document explains how to set up and use enhanced OpenTelemetry tracing with the LangGraph agent. The system supports multiple export options including Phoenix, file-based storage, console output, and OTLP collectors.

## Prerequisites

1. **Phoenix Server**: You need a Phoenix server running on port 6006
2. **Dependencies**: Install the LangGraph extras which include OpenTelemetry packages
3. **API Keys**: Set up your LLM provider API keys

## Installation

Install the required dependencies:

```bash
pip install tau-bench[langgraph]
```

This will install:
- `langgraph` and `langchain-core` for the LangGraph agent
- OpenTelemetry packages for tracing
- OpenInference instrumentation for LangChain and OpenAI

## Starting Phoenix Server

Make sure your Phoenix server is running on port 6006. The agent will automatically send traces to:
```
http://localhost:6006/v1/traces
```

## Usage

### Basic Phoenix Tracing (Default)

```python
from tau_bench.envs import get_env
from tau_bench.agents.langgraph_tool_call_agent import LangGraphToolCallingAgent

# Create environment
env = get_env(
    "retail",  # or "airline"
    user_strategy="llm",
    user_model="gpt-4o",
    user_provider="openai",
    task_split="test",
)

# Create agent with default Phoenix tracing
agent = LangGraphToolCallingAgent(
    tools_info=env.tools_info,
    wiki=env.wiki,
    model="gpt-4o",
    provider="openai",
    temperature=0.0,
    phoenix_port=6006,      # Phoenix server port (default)
    enable_tracing=True,    # Enable tracing (default)
)

# Run a task - traces will be sent to Phoenix
result = agent.solve(env=env, task_index=0, max_num_steps=30)
```

### Enhanced Tracing with File Export

```python
from tau_bench.agents.trace_exporters import TraceExportConfig

# Create custom trace configuration
trace_config = TraceExportConfig(
    phoenix_enabled=True,           # Send to Phoenix
    phoenix_port=6006,
    file_export_enabled=True,       # Also save to files
    file_export_path="./traces",    # Directory for trace files
    file_export_format="json",      # Format: "json" or "otlp"
    console_export_enabled=False,   # Optional: print to console
)

# Create agent with enhanced tracing
agent = LangGraphToolCallingAgent(
    tools_info=env.tools_info,
    wiki=env.wiki,
    model="gpt-4o",
    provider="openai",
    temperature=0.0,
    enable_tracing=True,
    trace_config=trace_config,      # Use custom config
)

# Run task - traces go to Phoenix AND files
result = agent.solve(env=env, task_index=0, max_num_steps=30)
```

### Using Trace Export Presets

```python
# Use predefined presets for common configurations
agent = LangGraphToolCallingAgent(
    tools_info=env.tools_info,
    wiki=env.wiki,
    model="gpt-4o",
    provider="openai",
    temperature=0.0,
    enable_tracing=True,
    trace_preset="phoenix_and_file",  # Available presets:
    # - "phoenix_only" (default)
    # - "file_only" 
    # - "phoenix_and_file"
    # - "full_export" (Phoenix + files + console)
    # - "otlp_collector" (OTLP + files)
)
```

### Disabling Tracing

If you want to disable tracing:

```python
agent = LangGraphToolCallingAgent(
    tools_info=env.tools_info,
    wiki=env.wiki,
    model="gpt-4o",
    provider="openai",
    temperature=0.0,
    enable_tracing=False,  # Disable tracing
)
```

### Custom Phoenix Port

If your Phoenix server is running on a different port:

```python
agent = LangGraphToolCallingAgent(
    tools_info=env.tools_info,
    wiki=env.wiki,
    model="gpt-4o",
    provider="openai",
    temperature=0.0,
    phoenix_port=8080,  # Custom port
)
```

## Running with the CLI

You can use the existing CLI with the LangGraph agent:

```bash
python run.py \
    --agent-strategy langgraph \
    --model gpt-4o \
    --model-provider openai \
    --env retail \
    --start-index 0 \
    --end-index 1
```

The tracing will be automatically enabled when using the `langgraph` agent strategy.

## What Gets Traced

The OpenTelemetry instrumentation captures:

### High-level spans:
- `langgraph_agent_solve`: The entire task execution
- `environment_reset`: Environment initialization
- `langgraph_execution`: LangGraph workflow execution

### Detailed spans:
- `llm_call`: Each LLM API call with model, provider, cost, and response details
- `tool_execution`: Tool execution with individual tool calls
- `tool_call_{name}`: Individual tool invocations with arguments and results
- `user_interaction`: User simulator interactions

### Span attributes include:
- Model and provider information
- Token costs and usage
- Tool names and arguments
- Response lengths and content
- Rewards and completion status
- Step counts and timing

## Testing

Run the test script to verify tracing is working:

```bash
python test_langgraph_tracing.py
```

This will:
1. Test the agent with tracing enabled
2. Test the agent with tracing disabled (fallback)
3. Report results and provide Phoenix UI link

## Viewing Traces in Phoenix

Once traces are being sent, you can view them in the Phoenix UI:

1. Open your browser to `http://localhost:6006`
2. Navigate to the traces section
3. You should see traces for each task execution with detailed spans

## Troubleshooting

### Missing Dependencies
If you get import errors, make sure you've installed the langgraph extras:
```bash
pip install tau-bench[langgraph]
```

### Phoenix Connection Issues
- Verify Phoenix is running on the correct port
- Check that `http://localhost:6006/v1/traces` is accessible
- Look for connection error messages in the logs

### No Traces Appearing
- Check the agent logs for OpenTelemetry initialization messages
- Verify the `enable_tracing=True` parameter
- Ensure your API keys are set correctly so tasks actually run

## Integration with Existing Code

The tracing is designed to be non-intrusive:
- If OpenTelemetry packages are not available, the agent falls back gracefully
- Existing code using the LangGraph agent will automatically get tracing
- No changes needed to existing task execution logic

The agent factory in `tau_bench/run.py` will automatically create LangGraph agents with tracing enabled when using the `langgraph` strategy.

## Trace Management and Export

### Managing Trace Files

Use the `trace_utils.py` script to manage and analyze saved traces:

```bash
# List all trace files
python trace_utils.py list

# Analyze a specific trace file
python trace_utils.py analyze traces/trace_2024-01-01T10-00-00.json

# Merge multiple trace files
python trace_utils.py merge traces/trace_*.json -o merged_traces.json

# Convert trace format (to OTLP or Jaeger)
python trace_utils.py convert trace.json trace_otlp.json --format otlp

# Export configuration template
python trace_utils.py config-template -o my_trace_config.json
```

### Available Export Formats

1. **JSON Format** (default): Human-readable trace format with full span details
2. **OTLP Format**: OpenTelemetry Protocol format for compatibility with OTLP collectors
3. **Console Output**: Real-time trace output to terminal (for debugging)

### Export Destinations

1. **Phoenix**: Real-time visualization and analysis
2. **File System**: Persistent storage for later analysis
3. **OTLP Collectors**: Integration with observability platforms (Jaeger, Zipkin, etc.)
4. **Console**: Development and debugging

### Trace Export Configuration Options

```python
from tau_bench.agents.trace_exporters import TraceExportConfig

config = TraceExportConfig(
    # Phoenix export
    phoenix_enabled=True,
    phoenix_port=6006,
    
    # File export
    file_export_enabled=True,
    file_export_path="./traces",
    file_export_format="json",  # or "otlp"
    
    # Console export (for debugging)
    console_export_enabled=False,
    
    # OTLP HTTP export (for collectors)
    otlp_http_enabled=False,
    otlp_http_endpoint="http://localhost:4318/v1/traces",
    
    # OTLP gRPC export
    otlp_grpc_enabled=False,
    otlp_grpc_endpoint="http://localhost:4317",
    
    # Custom attributes added to all spans
    custom_attributes={
        "experiment": "my_experiment",
        "version": "1.0"
    }
)
```

## Examples and Demos

Run the example script to see different tracing configurations:

```bash
python example_trace_export.py
```

This will demonstrate:
- Phoenix-only tracing (default)
- File-only export
- Combined Phoenix + file export
- Custom configurations
- OTLP collector integration
