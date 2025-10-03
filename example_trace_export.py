#!/usr/bin/env python3
"""
Example script demonstrating enhanced trace export capabilities in tau-bench.
Shows how to use different trace export configurations and presets.
"""

import os
import logging
from tau_bench.envs import get_env
from tau_bench.agents.langgraph_tool_call_agent import LangGraphToolCallingAgent
from tau_bench.agents.trace_exporters import TraceExportConfig, create_trace_export_presets

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_phoenix_only():
    """Example: Export traces only to Phoenix (default behavior)."""
    print("🔍 Example 1: Phoenix-only tracing")
    print("-" * 50)
    
    env = get_env(
        "retail",
        user_strategy="llm", 
        user_model="gpt-4o",
        user_provider="openai",
        task_split="test"
    )
    
    # Default behavior - traces go to Phoenix only
    agent = LangGraphToolCallingAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model="gpt-4o",
        provider="openai",
        temperature=0.0,
        enable_tracing=True,  # Default: Phoenix only
        phoenix_port=6006
    )
    
    print("Agent configured for Phoenix-only tracing")
    print("Traces will be sent to: http://localhost:6006")
    return agent, env


def example_file_export():
    """Example: Export traces to files only."""
    print("\n🔍 Example 2: File-only tracing")
    print("-" * 50)
    
    env = get_env(
        "retail",
        user_strategy="llm",
        user_model="gpt-4o", 
        user_provider="openai",
        task_split="test"
    )
    
    # Use preset for file-only export
    agent = LangGraphToolCallingAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model="gpt-4o",
        provider="openai",
        temperature=0.0,
        enable_tracing=True,
        trace_preset="file_only"  # Use preset
    )
    
    print("Agent configured for file-only tracing")
    print("Traces will be saved to: ./traces/")
    return agent, env


def example_phoenix_and_file():
    """Example: Export traces to both Phoenix and files."""
    print("\n🔍 Example 3: Phoenix + File tracing")
    print("-" * 50)
    
    env = get_env(
        "retail",
        user_strategy="llm",
        user_model="gpt-4o",
        user_provider="openai", 
        task_split="test"
    )
    
    # Use preset for Phoenix + file export
    agent = LangGraphToolCallingAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model="gpt-4o",
        provider="openai",
        temperature=0.0,
        enable_tracing=True,
        trace_preset="phoenix_and_file"
    )
    
    print("Agent configured for Phoenix + file tracing")
    print("Traces will be sent to Phoenix AND saved to files")
    return agent, env


def example_custom_config():
    """Example: Custom trace export configuration."""
    print("\n🔍 Example 4: Custom trace configuration")
    print("-" * 50)
    
    env = get_env(
        "retail",
        user_strategy="llm",
        user_model="gpt-4o",
        user_provider="openai",
        task_split="test"
    )
    
    # Create custom configuration
    custom_config = TraceExportConfig(
        phoenix_enabled=True,
        phoenix_port=6006,
        file_export_enabled=True,
        file_export_path="./custom_traces",
        file_export_format="json",
        console_export_enabled=True,  # Also print to console
        custom_attributes={
            "experiment": "custom_trace_demo",
            "version": "1.0"
        }
    )
    
    agent = LangGraphToolCallingAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model="gpt-4o",
        provider="openai",
        temperature=0.0,
        enable_tracing=True,
        trace_config=custom_config  # Use custom config
    )
    
    print("Agent configured with custom trace configuration")
    print("Traces will go to: Phoenix + ./custom_traces/ + console")
    return agent, env


def example_otlp_export():
    """Example: Export to OTLP collector."""
    print("\n🔍 Example 5: OTLP Collector export")
    print("-" * 50)
    
    env = get_env(
        "retail",
        user_strategy="llm",
        user_model="gpt-4o",
        user_provider="openai",
        task_split="test"
    )
    
    # Configuration for OTLP collector
    otlp_config = TraceExportConfig(
        phoenix_enabled=False,  # Disable Phoenix
        file_export_enabled=True,
        file_export_path="./otlp_traces",
        file_export_format="otlp",
        otlp_http_enabled=True,
        otlp_http_endpoint="http://localhost:4318/v1/traces"  # Standard OTLP endpoint
    )
    
    agent = LangGraphToolCallingAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model="gpt-4o",
        provider="openai",
        temperature=0.0,
        enable_tracing=True,
        trace_config=otlp_config
    )
    
    print("Agent configured for OTLP collector export")
    print("Traces will be sent to OTLP collector + saved as OTLP files")
    print("Note: Make sure your OTLP collector is running on localhost:4318")
    return agent, env


def run_task_example(agent, env, example_name: str):
    """Run a simple task to generate traces."""
    print(f"\n🚀 Running task for {example_name}...")
    
    try:
        result = agent.solve(
            env=env,
            task_index=0,  # First task
            max_num_steps=3  # Limit steps for demo
        )
        
        print(f"✅ Task completed!")
        print(f"   Reward: {result.reward}")
        print(f"   Cost: ${result.total_cost:.4f}")
        print(f"   Messages: {len(result.messages)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Task failed: {e}")
        return False


def show_available_presets():
    """Show all available trace export presets."""
    print("📋 Available Trace Export Presets:")
    print("=" * 50)
    
    presets = create_trace_export_presets()
    
    descriptions = {
        "phoenix_only": "Export traces only to Phoenix (default)",
        "file_only": "Export traces only to JSON files", 
        "phoenix_and_file": "Export to both Phoenix and files",
        "full_export": "Export to Phoenix, files, and console",
        "otlp_collector": "Export to OTLP collector and files"
    }
    
    for preset_name, config in presets.items():
        desc = descriptions.get(preset_name, "Custom preset")
        print(f"\n🔧 {preset_name}")
        print(f"   Description: {desc}")
        print(f"   Phoenix: {'✅' if config.phoenix_enabled else '❌'}")
        print(f"   File export: {'✅' if config.file_export_enabled else '❌'}")
        print(f"   Console: {'✅' if config.console_export_enabled else '❌'}")
        print(f"   OTLP HTTP: {'✅' if config.otlp_http_enabled else '❌'}")


def main():
    """Run all trace export examples."""
    print("🎯 Tau-Bench Enhanced Trace Export Examples")
    print("=" * 60)
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable is required")
        print("   Set it with: export OPENAI_API_KEY='your-api-key'")
        return
    
    # Show available presets
    show_available_presets()
    
    # Run examples (you can comment out examples you don't want to run)
    examples = [
        ("Phoenix Only", example_phoenix_only),
        ("File Export", example_file_export), 
        ("Phoenix + File", example_phoenix_and_file),
        ("Custom Config", example_custom_config),
        # ("OTLP Export", example_otlp_export),  # Uncomment if you have OTLP collector
    ]
    
    for example_name, example_func in examples:
        try:
            agent, env = example_func()
            
            # Ask user if they want to run this example
            response = input(f"\nRun {example_name} example? (y/n/q): ").lower().strip()
            
            if response == 'q':
                print("Exiting...")
                break
            elif response == 'y':
                success = run_task_example(agent, env, example_name)
                if success:
                    print(f"✅ {example_name} example completed successfully")
                else:
                    print(f"❌ {example_name} example failed")
            else:
                print(f"⏭️  Skipping {example_name} example")
                
        except Exception as e:
            print(f"❌ Error in {example_name} example: {e}")
    
    print("\n🎉 Trace export examples completed!")
    print("\nNext steps:")
    print("1. Check your trace files in ./traces/ or ./custom_traces/")
    print("2. View traces in Phoenix UI at http://localhost:6006")
    print("3. Use trace_utils.py to analyze and manage trace files")
    print("4. Run: python trace_utils.py list")


if __name__ == "__main__":
    main()
