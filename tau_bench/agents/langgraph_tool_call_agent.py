# Copyright Sierra

import json
from typing import List, Optional, Dict, Any, TypedDict, Annotated, Literal
from functools import partial
import os

# LangChain OpenAI imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME

from phoenix.otel import register
from opentelemetry import trace

tracer_provider = register(
    project_name="tau-bench", # sets a project name for spans
    batch=True, # uses a batch span processor
    auto_instrument=True, # uses all installed OpenInference instrumentors
    protocol="http/protobuf",
    endpoint="http://localhost:6006/v1/traces",
    
)

# Get a tracer for manual span creation
tracer = trace.get_tracer(__name__)


import logging
import os
logging.basicConfig(level=logging.INFO)

## suppress openai logging
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
LANGGRAPH_AVAILABLE = True


class OTELFileExporter:
    """Custom OpenTelemetry span exporter that saves traces in OTEL format to local files."""
    
    def __init__(self, file_path: str):
        """Initialize the OTEL file exporter."""
        self.file_path = file_path
        self.trace_spans = {}  # trace_id -> list of spans
        
    def export(self, spans):
        """Export spans to OTEL format files, grouped by trace_id."""
        import json
        import os
        from datetime import datetime
        
        try:
            # Group spans by trace_id
            for span in spans:
                trace_id = format(span.context.trace_id, '032x')
                
                if trace_id not in self.trace_spans:
                    self.trace_spans[trace_id] = []
                
                self.trace_spans[trace_id].append(span)
            
            # Don't export during regular export() calls - only accumulate spans
            # We'll only export complete traces during force_flush() or shutdown()
            # This prevents exporting incomplete traces when BatchSpanProcessor calls export()
            
        except Exception as e:
            logging.error(f"Failed to export OTEL spans: {e}")
    
    def _export_otel_trace(self, trace_id: str, spans):
        """Export a complete trace in OTEL format."""
        import json
        import os
        from datetime import datetime
        
        try:
            # Create OTEL format data
            otel_data = {
                "resourceSpans": [{
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": "tau-bench"}},
                            {"key": "service.version", "value": {"stringValue": "0.1.0"}},
                            {"key": "trace.id", "value": {"stringValue": trace_id}}
                        ]
                    },
                    "scopeSpans": [{
                        "scope": {
                            "name": "tau_bench.agents.langgraph_tool_call_agent",
                            "version": "0.1.0"
                        },
                        "spans": []
                    }]
                }]
            }
            
            # Convert spans to OTEL format
            for span in spans:
                otel_span = {
                    "traceId": format(span.context.trace_id, '032x'),
                    "spanId": format(span.context.span_id, '016x'),
                    "parentSpanId": format(span.parent.span_id, '016x') if span.parent else "",
                    "name": span.name,
                    "kind": 1,  # SPAN_KIND_INTERNAL
                    "startTimeUnixNano": str(span.start_time),
                    "endTimeUnixNano": str(span.end_time) if span.end_time else str(span.start_time),
                    "attributes": [],
                    "status": {
                        "code": span.status.status_code.value if span.status else 1,
                        "message": span.status.description if span.status else ""
                    }
                }
                
                # Add attributes
                if span.attributes:
                    for key, value in span.attributes.items():
                        otel_span["attributes"].append({
                            "key": key,
                            "value": {"stringValue": str(value)}
                        })
                
                # Add events
                if span.events:
                    otel_span["events"] = []
                    for event in span.events:
                        otel_event = {
                            "timeUnixNano": str(event.timestamp),
                            "name": event.name,
                            "attributes": []
                        }
                        if event.attributes:
                            for key, value in event.attributes.items():
                                otel_event["attributes"].append({
                                    "key": key,
                                    "value": {"stringValue": str(value)}
                                })
                        otel_span["events"].append(otel_event)
                
                otel_data["resourceSpans"][0]["scopeSpans"][0]["spans"].append(otel_span)
            
            # Generate filename
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"otel_trace_{trace_id}_{timestamp_str}.json"
            filepath = os.path.join(self.file_path, filename)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(otel_data, f, indent=2, default=str)
            
            logging.info(f"OTEL trace saved: {filepath}")
            
        except Exception as e:
            logging.error(f"Failed to export OTEL trace {trace_id}: {e}")
    
    def shutdown(self):
        """Shutdown the exporter and flush any remaining traces."""
        self.force_flush()
    
    def force_flush(self, timeout_millis: int = 30000):
        """Force flush any pending spans."""
        # Export all accumulated traces
        for trace_id, spans in list(self.trace_spans.items()):
            # Prioritize traces with the main agent span (complete traces)
            has_main_agent_span = any(span.name == "langgraph_agent_solve" for span in spans)
            
            if has_main_agent_span:
                # This is a complete trace with the main agent execution
                self._export_otel_trace(trace_id, spans)
            elif len(spans) > 3:  # Only export standalone traces if they have substantial content
                # This might be a trace without the main span but with meaningful content
                self._export_otel_trace(trace_id, spans)
            # Skip traces that are just user simulator calls (typically 1-3 spans)
            
            del self.trace_spans[trace_id]
        return True


# State definition for LangGraph
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    data: Dict[str, Any]
    current_step: int
    max_steps: int
    total_cost: float
    done: bool
    reward: float
    info: Dict[str, Any]


def convert_tools_to_openai_format(tools_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert tools info to OpenAI function calling format"""
    # tools_info already contains the correct OpenAI format from tool.get_info()
    # Each tool already has {"type": "function", "function": {...}}
    return tools_info


def call_model(state: AgentState, model: str, provider: str, temperature: float, tools_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """LangGraph Agent Node: Generate LLM response"""
    with tracer.start_as_current_span(
        "llm_call",
        attributes={
            "llm.model": model,
            "llm.provider": provider,
            "llm.temperature": temperature,
            "llm.step": state["current_step"],
            "llm.input_messages": len(state["messages"])
        }
    ) as llm_span:
        # Create ChatOpenAI client
        openai_client = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_base=os.getenv("AGENT_BASE_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
            extra_body={"budget": 8, "return_response_only": False}
        )
        
        # Convert tools to OpenAI format
        openai_tools = convert_tools_to_openai_format(tools_info) if tools_info else None
        
        # Generate response using LangChain ChatOpenAI
        if openai_tools:
            response_message = openai_client.invoke(
                state["messages"],
                tools=openai_tools
            )
        else:
            response_message = openai_client.invoke(state["messages"])
        
        logging.info(f"Response message: {response_message}")
        
        # Note: Cost tracking is disabled when using LangChain ChatOpenAI
        # To enable cost tracking, consider using LangChain's callback handlers
        # or integrating with services like LangSmith
        cost = 0.0
        
        # MATCH ORIGINAL FLOW: Extract action from full response, then truncate for storage
        # This matches the original agent's flow: message_to_action() -> env.step() -> truncate -> store
        
        # LangChain returns an AIMessage directly, so we need to handle it differently
        # Check if the response has tool calls
        if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
            # Store original tool calls count for tracing
            original_tool_count = len(response_message.tool_calls)
            
            # MATCH ORIGINAL: Only keep first tool call for conversation history
            # This matches line 180 in original: next_message["tool_calls"] = next_message["tool_calls"][:1]
            first_tool_call = response_message.tool_calls[0]
            
            # LangChain tool calls are already in the right format
            tool_calls = [first_tool_call]
            
            ai_message = AIMessage(
                content=response_message.content or "",
                tool_calls=tool_calls
            )
            llm_span.set_attributes({
                "llm.tool_calls_count": original_tool_count,
                "llm.tool_calls_kept": 1,
                "llm.tool_names": [first_tool_call["name"]]
            })
        else:
            ai_message = AIMessage(content=response_message.content or "")
        
        # Add span attributes
        llm_span.set_attributes({
            "llm.response_cost": cost,
            "llm.response_length": len(response_message.content or ""),
            "llm.has_tool_calls": bool(hasattr(response_message, 'tool_calls') and response_message.tool_calls)
        })
        
        return {
            "messages": [ai_message],
            "total_cost": state["total_cost"] + cost,


            # MATCH ORIGINAL: Increment step on each LLM call (each loop iteration)
            "current_step": state["current_step"] + 1
        }


def call_tools(state: AgentState, env: Env) -> Dict[str, Any]:
    """LangGraph Tools Node: Execute tool calls via environment"""
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}
    
    with tracer.start_as_current_span(
        "tool_execution",
        attributes={
            "tools.count": len(last_message.tool_calls),
            "tools.step": state["current_step"]
        }
    ) as tools_span:
        env_response = None
        
        # Process the first (and now only) tool call since truncation happened in call_model
        tc = last_message.tool_calls[0]
        name = tc["name"]
        args = tc["args"]
        
        with tracer.start_as_current_span(
            f"tool_call_{name}",
            attributes={
                "tool.name": name,
                "tool.args": str(args),
                "tool.call_index": 0
            }
        ) as tool_span:
            action = Action(name=name, kwargs=args)
            env_response = env.step(action)
            
            tool_span.set_attributes({
                "tool.response_length": len(env_response.observation),
                "tool.reward": env_response.reward,
                "tool.done": env_response.done
            })
        
        tool_msg = ToolMessage(
            tool_call_id=tc["id"],
            name=name,
            content=env_response.observation
        )
        
        msgs = [tool_msg]
        
        # Add overall tool execution attributes
        tools_span.set_attributes({
            "tools.final_reward": env_response.reward if env_response else 0.0,
            "tools.final_done": env_response.done if env_response else False,
            "tools.messages_created": len(msgs)
        })
        
        result = {"messages": msgs}
        if env_response:
            result.update({
                "reward": env_response.reward,
                "done": env_response.done,
                "info": {**state["info"], **env_response.info.model_dump()},
                # Step increment moved to call_model to match original loop behavior
            })
        
        return result


def call_user(state: AgentState, env: Env) -> Dict[str, Any]:
    """LangGraph User Node: Handle user interaction via environment"""
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage):
        return {}
    
    with tracer.start_as_current_span(
        "user_interaction",
        attributes={
            "user.step": state["current_step"],
            "user.message_length": len(last_message.content or "")
        }
    ) as user_span:
        content = last_message.content or ""
        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
        env_response = env.step(action)
        user_msg = HumanMessage(content=env_response.observation)
        
        user_span.set_attributes({
            "user.response_length": len(env_response.observation),
            "user.reward": env_response.reward,
            "user.done": env_response.done
        })
        
        return {
            "messages": [user_msg],
            "reward": env_response.reward,
            "done": env_response.done,
            "info": {**state["info"], **env_response.info.model_dump()},
            # Step increment moved to call_model to match original loop behavior
        }


def should_continue(state: AgentState) -> Literal["tools", "user", END]:
    """Determine the next step based on current state - matches original agent behavior"""
    # MATCH ORIGINAL: Only check done flag. Step limit is handled by LangGraph recursion_limit
    # Original only has: if env_response.done: break
    if state["done"]:
        return END
    
    last_message = state["messages"][-1]
    
    if isinstance(last_message, AIMessage):
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        else:
            return "user"
    
    return END


class LangGraphToolCallingAgent(Agent):
    """
    LangGraph-based tool calling agent that uses a state graph to manage
    conversation flow, tool execution, and response generation.
    
    This agent uses LangChain's ChatOpenAI client for LLM interactions and
    LangGraph's structured approach with nodes and edges for better control flow.
    
    To use this agent, install the optional dependencies:
    pip install tau-bench[langgraph]
    
    Or manually:
    pip install langgraph langchain-core langchain-openai
    """
    
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        save_traces_locally: bool = False,
        local_traces_path: str = "./traces",
    ):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is required but not available. Please install with: "
                "pip install tau-bench[langgraph] or pip install langgraph langchain-core"
            )
        
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.save_traces_locally = save_traces_locally
        self.local_traces_path = local_traces_path
        
        # Add file exporter if local saving is enabled
        if save_traces_locally:
            self._setup_local_trace_saving()
    
    def _setup_local_trace_saving(self):
        """Set up OTEL format trace saving by adding a custom span processor."""
        import os
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        
        # Create traces directory if it doesn't exist
        os.makedirs(self.local_traces_path, exist_ok=True)
        
        # Add our custom OTEL exporter to the existing tracer provider
        otel_exporter = OTELFileExporter(self.local_traces_path)
        tracer_provider.add_span_processor(BatchSpanProcessor(otel_exporter))
        
        logging.info(f"OTEL trace saving enabled. Traces will be saved to: {self.local_traces_path}")
    
    def _create_graph(self, env: Env, max_num_steps: int = 30) -> StateGraph:
        """Create the LangGraph state graph"""
        # Create the graph
        workflow = StateGraph(AgentState)
            
        # Add nodes
        workflow.add_node("agent", partial(call_model, 
                                         model=self.model, 
                                         provider=self.provider, 
                                         temperature=self.temperature, 
                                         tools_info=self.tools_info))
        
        workflow.add_node("tools", partial(call_tools, env=env))
        workflow.add_node("user", partial(call_user, env=env))
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "user": "user",
                END: END,
            },
        )
        
        # Add edges back to agent
        workflow.add_edge("tools", "agent")
        workflow.add_edge("user", "agent")
        
        return workflow.compile()

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30, memory: Optional[Any] = None, mode: str = "train", budget: int = 4
    ) -> SolveResult:
        """Solve a single task"""
        # Create a single parent span for the entire graph invocation
        with tracer.start_as_current_span(
            "langgraph_agent_solve",
            attributes={
                "agent.model": self.model,
                "agent.provider": self.provider,
                "agent.temperature": self.temperature,
                "agent.max_steps": max_num_steps,
                "task.index": task_index if task_index is not None else -1,
                "env.name": getattr(env, 'name', 'unknown'),
            }
        ) as solve_span:
            # Reset environment
            with tracer.start_as_current_span("environment_reset") as reset_span:
                env_reset_res = env.reset(task_index=task_index)
                obs = env_reset_res.observation
                info = env_reset_res.info.model_dump()
                reset_span.set_attributes({
                    "env.observation_length": len(obs),
                    "env.info": str(info)  # Truncate long info
                })
            
            # Initialize state
            initial_state: AgentState = {
                "messages": [
                    SystemMessage(content=self.wiki),
                    HumanMessage(content=obs)
                ],
                "data": env.data,
                "current_step": 0,
                "max_steps": max_num_steps,
                "total_cost": 0.0,
                "done": False,
                "reward": 0.0,
                "info": info
            }
            
            # Create and run the graph within a span
            with tracer.start_as_current_span("langgraph_execution") as graph_span:
                graph = self._create_graph(env, max_num_steps)
                final_state = graph.invoke(
                    initial_state, 
                    config={"recursion_limit": max_num_steps}  # MATCH ORIGINAL: Same step limit
                )
                
                # Add final state attributes to the graph span
                graph_span.set_attributes({
                    "graph.final_step": final_state["current_step"],
                    "graph.total_cost": final_state["total_cost"],
                    "graph.final_reward": final_state["reward"],
                    "graph.done": final_state["done"],
                    "graph.message_count": len(final_state["messages"])
                })
            
            # Add final attributes to the solve span
            solve_span.set_attributes({
                "solve.final_step": final_state["current_step"],
                "solve.total_cost": final_state["total_cost"],
                "solve.final_reward": final_state["reward"],
                "solve.done": final_state["done"],
                "solve.message_count": len(final_state["messages"])
            })
            
            return self._build_solve_result(final_state)
    
    def _build_solve_result(self, final_state: AgentState) -> SolveResult:
        """Convert final state to SolveResult format - matches original agent exactly"""
        # Convert messages back to the expected format
        messages = []
        for msg in final_state["messages"]:
            if isinstance(msg, SystemMessage):
                messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                msg_dict = {"role": "assistant", "content": msg.content}
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    # MATCH ORIGINAL BEHAVIOR: Ensure tool calls are in the right format
                    # The original agent already limits to first tool call, so we should have only one
                    tool_calls = []
                    for tc in msg.tool_calls:
                        if isinstance(tc, dict):
                            # LangChain format: {"name": "func_name", "args": {...}, "id": "..."}
                            tool_calls.append({
                                "id": tc.get("id", ""),
                                "function": {
                                    "name": tc.get("name", ""),
                                    "arguments": json.dumps(tc.get("args", {}))
                                },
                                "type": "function"
                            })
                        else:
                            # Handle object format (LangChain ToolCall objects)
                            tool_calls.append({
                                "id": getattr(tc, "id", ""),
                                "function": {
                                    "name": getattr(tc, "name", ""),
                                    "arguments": json.dumps(getattr(tc, "args", {}))
                                },
                                "type": "function"
                            })
                    msg_dict["tool_calls"] = tool_calls
                messages.append(msg_dict)
            elif isinstance(msg, ToolMessage):
                messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": getattr(msg, "tool_call_id", ""),
                    "name": getattr(msg, "name", "")
                })
        
        return SolveResult(
            reward=final_state["reward"],
            info=final_state["info"],
            messages=messages,
            total_cost=final_state["total_cost"],
        )



def message_to_action(
    message: Dict[str, Any],
) -> Action:
    """Legacy function for compatibility - converts message to action"""
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})