#!/usr/bin/env python3
"""
Example showing how to use the LangGraph agent with local trace saving.
"""

import os
import logging
from tau_bench.envs import get_env
from tau_bench.agents.langgraph_tool_call_agent import LangGraphToolCallingAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_with_local_traces():
    """Example of running the agent with both Phoenix and local trace saving."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        return False
    
    try:
        # Create environment
        logger.info("Creating retail environment...")
        env = get_env(
            "retail",
            user_strategy="llm",
            user_model="gpt-4o",
            user_provider="openai",
            task_split="test",
        )
        
        # Create LangGraph agent with OTEL trace saving enabled
        logger.info("Creating LangGraph agent with OTEL trace saving...")
        agent = LangGraphToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model="gpt-4o",
            provider="openai",
            temperature=0.0,
            save_traces_locally=True,        # Enable OTEL format saving
            local_traces_path="./my_traces", # Custom path (optional)
        )
        
        # Run a task
        logger.info("Running task - traces will be saved both to Phoenix AND locally in OTEL format...")
        result = agent.solve(
            env=env,
            task_index=0,
            max_num_steps=5,
        )
        
        logger.info(f"✅ Task completed!")
        logger.info(f"   - Reward: {result.reward}")
        logger.info(f"   - Cost: {result.total_cost}")
        logger.info(f"   - Messages: {len(result.messages)}")
        
        logger.info("\n📁 Check the './my_traces' directory for OTEL format trace files")
        logger.info("🌐 Also check Phoenix UI at http://localhost:6006")
        
        # Show what files were created
        import os
        if os.path.exists("./my_traces"):
            trace_files = [f for f in os.listdir("./my_traces") if f.startswith("otel_trace_")]
            logger.info(f"📄 Created {len(trace_files)} OTEL trace file(s): {trace_files}")
            
            # Show a sample of what the OTEL format looks like
            if trace_files:
                sample_file = os.path.join("./my_traces", trace_files[0])
                try:
                    import json
                    with open(sample_file, 'r') as f:
                        otel_data = json.load(f)
                    logger.info(f"📋 OTEL format preview - spans in trace: {len(otel_data['resourceSpans'][0]['scopeSpans'][0]['spans'])}")
                except Exception as e:
                    logger.warning(f"Could not preview OTEL file: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def example_default_path():
    """Example using the default traces path."""
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        return False
    
    try:
        env = get_env("retail", user_strategy="llm", user_model="gpt-4o", user_provider="openai", task_split="test")
        
        # Using default path (./traces)
        agent = LangGraphToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model="gpt-4o",
            provider="openai",
            temperature=0.0,
            save_traces_locally=True,  # Uses default path: ./traces
        )
        
        result = agent.solve(env=env, task_index=0, max_num_steps=3)
        
        logger.info(f"✅ Task completed with reward: {result.reward}")
        logger.info("📁 Check the './traces' directory for saved trace files")
        
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return False

if __name__ == "__main__":
    logger.info("🧪 Testing local trace saving...")
    logger.info("=" * 60)
    
    logger.info("Example 1: Custom traces directory")
    success1 = example_with_local_traces()
    
    logger.info("\n" + "=" * 60)
    
    logger.info("Example 2: Default traces directory")
    success2 = example_default_path()
    
    logger.info("\n" + "=" * 60)
    
    if success1 and success2:
        logger.info("✅ All examples completed!")
        logger.info("Check both Phoenix UI and local trace files.")
    else:
        logger.error("❌ Some examples failed.")
