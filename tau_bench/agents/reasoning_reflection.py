# Copyright Sierra

from typing import Dict, Any, List, Optional
from litellm import completion
import json
import os


class ReasoningReflectionGenerator:
    """
    Generates reasoning reflections after tool call iterations to summarize
    the information retrieved and assess its importance to the original problem.
    
    Supports two modes:
    1. Separate LLM call mode (original)
    2. Inline reflection mode (new) - adds <tool_reflection> prompts
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        temperature: float = 0.0,
        use_inline_reflection: bool = True
    ):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.use_inline_reflection = use_inline_reflection
    
    def generate_reflection(
        self,
        original_task: str,
        tool_call_info: Dict[str, Any],
        tool_response: str,
        current_context: Optional[str] = None
    ) -> str:
        """
        Generate a reasoning reflection based on the tool call and its result.
        
        Args:
            original_task: The original problem/task the agent is trying to solve
            tool_call_info: Information about the tool call (name, arguments)
            tool_response: The response/output from the tool call
            current_context: Optional context about the current state of problem solving
            
        Returns:
            A formatted reasoning reflection string or inline prompt
        """
        
        if self.use_inline_reflection:
            # For inline reflection, we still need to make an LLM call to generate the reflection
            prompt = self._create_inline_reflection_prompt(
                original_task, tool_call_info, tool_response, current_context
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
                return reflection_content  # Return the generated reflection directly
                
            except Exception as e:
                # Fallback to a simple template-based reflection if LLM call fails
                return f"<tool_reflection>Tool call to {tool_call_info.get('name', 'Unknown')} completed. Information retrieved may be relevant to the task.</tool_reflection>"
        else:
            # Original separate LLM call approach
            prompt = self._create_reflection_prompt(
                original_task, tool_call_info, tool_response, current_context
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
                return f"<reasoning_reflection>\n{reflection_content}\n</reasoning_reflection>"
                
            except Exception as e:
                # Fallback to a simple template-based reflection if LLM call fails
                return self._create_fallback_reflection(tool_call_info, tool_response)
    
    def _create_reflection_prompt(
        self,
        original_task: str,
        tool_call_info: Dict[str, Any],
        tool_response: str,
        current_context: Optional[str] = None
    ) -> str:
        """Create the prompt for generating the reasoning reflection."""
        
        context_section = f"\n\nCurrent Context: {current_context}" if current_context else ""
        
        return f"""You are an AI assistant that generates concise reasoning reflections after tool calls. 

Your task is to analyze a tool call and its result, then provide a brief summary that includes:
1. What information was retrieved by the tool
2. How important/relevant this information is to solving the original problem
3. What insights or next steps this suggests

Original Task/Problem: {original_task}

Tool Call Made:
- Tool Name: {tool_call_info.get('name', 'Unknown')}
- Arguments: {json.dumps(tool_call_info.get('arguments', {}), indent=2)}

Tool Response/Output:
{tool_response}
{context_section}

Generate a concise reasoning reflection (2-4 sentences) that summarizes:
1. The key information obtained from this tool call
2. Its relevance/importance to the original problem (High/Medium/Low)
3. Any insights or implications for next steps

Keep it brief and focused. Do not include the <reasoning_reflection> tags in your response."""

    def _create_inline_reflection_prompt(
        self,
        original_task: str,
        tool_call_info: Dict[str, Any],
        tool_response: str,
        current_context: Optional[str] = None
    ) -> str:
        """Create an inline reflection prompt for generating reflection content."""
        
        tool_name = tool_call_info.get('name', 'Unknown')
        context_section = f" Context: {current_context}" if current_context else ""
        
        return f"""You are an AI assistant analyzing a tool call and its result. Generate a brief reflection on what was learned and its relevance to the original task.

Original Task: {original_task}

Tool Used: {tool_name}
Tool Arguments: {json.dumps(tool_call_info.get('arguments', {}), indent=2)}
Tool Response: {tool_response}
{context_section}

Generate a concise reflection (2-3 sentences) that:
1. Summarizes the key information obtained
2. Assesses its relevance to the original task (High/Medium/Low relevance)
3. Notes any insights for next steps

Format your response as: <tool_reflection>Your reflection here</tool_reflection>"""

    def _create_fallback_reflection(
        self,
        tool_call_info: Dict[str, Any],
        tool_response: str
    ) -> str:
        """Create a simple fallback reflection if LLM generation fails."""
        
        tool_name = tool_call_info.get('name', 'Unknown')
        response_preview = tool_response[:100] + "..." if len(tool_response) > 100 else tool_response
        
        return f"""<reasoning_reflection>
Tool call to {tool_name} completed. Retrieved information: {response_preview}. 
This information contributes to understanding the current problem state and may inform subsequent actions.
</reasoning_reflection>"""

    def should_generate_reflection(
        self,
        tool_call_info: Dict[str, Any],
        iteration_count: int
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
