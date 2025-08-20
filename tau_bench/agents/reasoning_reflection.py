# Copyright Sierra

from typing import Dict, Any, List, Optional
from litellm import completion
import json
import os


class ReasoningReflectionGenerator:
    """
    Generates reasoning reflections after tool call iterations to summarize
    the information retrieved and assess its importance to the original problem.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        temperature: float = 0.0
    ):
        self.model = model
        self.provider = provider
        self.temperature = temperature
    
    def generate_reflection(
        self,
        messages: List[Dict[str, Any]],
        tool_call_info: Dict[str, Any],
        tool_response: str
    ) -> str:
        """
        Generate a reasoning reflection based on the tool call and its result.
        
        Args:
            messages: Full conversation history for context
            tool_call_info: Information about the tool call (name, arguments)
            tool_response: The response/output from the tool call
            
        Returns:
            A first-person reflection from the assistant's perspective
        """
        
        # Generate reflection content for assistant message
        prompt = self._create_first_person_reflection_prompt(
            messages, tool_call_info, tool_response
        )
        
        try:
            # Handle vLLM with custom base URL
            if self.provider == "hosted_vllm":
                response = completion(
                    model=self.model,
                    custom_llm_provider=self.provider,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    base_url=os.getenv("VLLM_BASE_URL"),
                    api_key=os.getenv("VLLM_API_KEY"),
                )
            else:
                response = completion(
                    model=self.model,
                    custom_llm_provider=self.provider,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
            
            reflection_content = response.choices[0].message.content.strip()
            return reflection_content  # Return clean reflection content for assistant message
            
        except Exception as e:
            raise e
    
    def _create_first_person_reflection_prompt(
        self,
        messages: List[Dict[str, Any]],
        tool_call_info: Dict[str, Any],
        tool_response: str
    ) -> str:
        """Create the prompt for generating first-person reflection content."""
        
        # Format the conversation history for context
        conversation_context = ""
        for msg in messages[-10:]:  # Use last 10 messages for context
            role = msg.get('role', 'unknown')
            content = msg.get('content', '') or ''  # Handle None content
            if role == 'system':
                continue  # Skip system messages in context
            elif role == 'user':
                conversation_context += f"User: {content[:200]}...\n" if len(content) > 200 else f"User: {content}\n"
            elif role == 'assistant':
                conversation_context += f"Me: {content[:200]}...\n" if len(content) > 200 else f"Me: {content}\n"
            elif role == 'tool':
                conversation_context += f"Tool Result: {content[:100]}...\n" if len(content) > 100 else f"Tool Result: {content}\n"
        
        return f"""You are an AI assistant reflecting on your own actions. Based on the conversation history and the tool call you just made, provide a brief reflection on what you learned and how it helps you solve the user's problem.

Recent Conversation:
{conversation_context}

Tool Call I Just Made:
- Tool: {tool_call_info.get('name', 'Unknown')}
- Arguments: {json.dumps(tool_call_info.get('arguments', {}), indent=2)}

Tool Response I Received:
{tool_response}

Provide a brief first-person reflection (2-3 sentences) from your perspective as the assistant. Focus on:
1. What key information you obtained from this tool call
2. How this information helps you progress toward solving the user's request
3. What you plan to do next (if applicable)

You are the assistant reflecting on your own action. Be concise, natural and conversational."""


    def should_generate_reflection(
        self,
        tool_call_info: Dict[str, Any]
    ) -> bool:
        """
        Determine if a reflection should be generated for this tool call.
        Can be used to skip reflections for certain tool types or conditions.
        """
        
        # Skip reflection for respond actions (final responses)
        if tool_call_info.get('name') == 'respond':
            return False
            
        # Skip reflection for think actions (internal reasoning)
        if tool_call_info.get('name') == 'think':
            return False
            
        # Always generate for other tool calls
        return True


def extract_tool_call_info(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract tool call information from a message.
    
    Args:
        message: The message containing tool call information
        
    Returns:
        Dictionary with tool call info or None if no tool call found
    """
    
    if "tool_calls" in message and message["tool_calls"]:
        tool_call = message["tool_calls"][0]  # Take first tool call
        return {
            "name": tool_call["function"]["name"],
            "arguments": json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
        }
    
    return None


def extract_original_task(messages: List[Dict[str, Any]]) -> str:
    """
    Extract the original task/problem from the message history.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        The original task description
    """
    
    # Look for the first user message which typically contains the task
    for message in messages:
        if message.get("role") == "user":
            return message.get("content", "Unknown task")
    
    return "Unknown task"
