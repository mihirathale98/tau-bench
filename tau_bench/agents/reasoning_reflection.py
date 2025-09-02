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
            A strategic reflection from the assistant's perspective
        """
        
        # Extract original goal for context
        original_goal = self._extract_user_goal(messages)
        
        # Generate reflection content for assistant message
        prompt = self._create_strategic_reflection_prompt(
            messages, tool_call_info, tool_response, original_goal
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
            # Fallback to template-based reflection if LLM fails
            return self._generate_fallback_reflection(tool_call_info, tool_response)
    
    def _create_strategic_reflection_prompt(
        self,
        messages: List[Dict[str, Any]],
        tool_call_info: Dict[str, Any],
        tool_response: str,
        original_goal: str
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
        
        # Detect potential inefficiencies
        issues = self._detect_inefficiencies(messages, tool_response)
        issues_text = "\n".join([f"⚠️ {issue}" for issue in issues]) if issues else "No obvious issues detected."
        
        return f"""You are an AI assistant critically analyzing your own performance. Based on the conversation history and the tool call you just made, provide a strategic reflection that helps improve your decision-making.

**USER'S ORIGINAL GOAL:** {original_goal}

**POTENTIAL ISSUES DETECTED:**
{issues_text}

Recent Conversation:
{conversation_context}

Tool Call I Just Made:
- Tool: {tool_call_info.get('name', 'Unknown')}
- Arguments: {json.dumps(tool_call_info.get('arguments', {}), indent=2)}

Tool Response I Received:
{tool_response}

Provide a strategic reflection (2-4 sentences) that includes:

**CRITICAL ANALYSIS:**
- Was this tool call efficient? Did it move me closer to the goal?
- Are there any red flags, errors, or inconsistencies in the response?
- Am I missing any key information or making assumptions?

**STRATEGIC PLANNING:**
- What is the most efficient next step to achieve the user's goal?
- Are there any shortcuts or better approaches I should consider?
- What potential issues should I watch out for?

Be direct, strategic, and focus on improving efficiency rather than just summarizing what happened."""


    def should_generate_reflection(
        self,
        tool_call_info: Dict[str, Any],
        messages: Optional[List[Dict[str, Any]]] = None,
        tool_response: Optional[str] = None
    ) -> bool:
        """
        Determine if a reflection should be generated for this tool call.
        Uses smart filtering to reduce unnecessary reflections.
        
        Args:
            tool_call_info: Information about the tool call
            messages: Conversation history for context analysis
            tool_response: The tool's response for success/failure detection
        """
        
        # Skip reflection for respond actions (final responses)
        if tool_call_info.get('name') == 'respond':
            return False
            
        # Skip reflection for think actions (internal reasoning)
        if tool_call_info.get('name') == 'think':
            return False
        
        # If we don't have context, use conservative approach (generate reflection)
        if not messages or not tool_response:
            return True
            
        # Smart filtering based on multiple criteria
        return self._should_reflect_smart_filter(tool_call_info, messages, tool_response)
    
    def _detect_inefficiencies(self, messages: List[Dict[str, Any]], tool_response: str) -> List[str]:
        """Detect potential inefficiencies or issues in the conversation."""
        issues = []
        
        # Count failed tool calls
        failed_calls = 0
        for msg in messages[-5:]:  # Check last 5 messages
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if "error" in content.lower() or "not found" in content.lower() or "invalid" in content.lower():
                    failed_calls += 1
        
        if failed_calls >= 2:
            issues.append("Multiple failed tool calls detected - consider alternative approach")
        
        # Check for repetitive patterns
        tool_calls = []
        for msg in messages[-10:]:
            if msg.get("tool_calls"):
                tool_name = msg["tool_calls"][0]["function"]["name"]
                tool_calls.append(tool_name)
        
        if len(tool_calls) > 3 and len(set(tool_calls[-3:])) == 1:
            issues.append("Repetitive tool calls detected - may be stuck in a loop")
        
        return issues
    
    def _extract_user_goal(self, messages: List[Dict[str, Any]]) -> str:
        """Extract the user's original goal from the conversation."""
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", "")
                if content and len(content) > 20:  # Skip very short messages
                    return content[:300] + "..." if len(content) > 300 else content
        return "Help the user with their request"
    
    def _generate_fallback_reflection(self, tool_call_info: Dict[str, Any], tool_response: str) -> str:
        """Generate a simple fallback reflection when LLM fails."""
        tool_name = tool_call_info.get('name', 'Unknown')
        return f"<reasoning_reflection>\nExecuted {tool_name} tool call. Need to analyze if this moved me closer to the user's goal and determine the most efficient next step.\n</reasoning_reflection>"


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
