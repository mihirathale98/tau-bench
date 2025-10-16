# Copyright Sierra

import json
from litellm import completion
import litellm
from typing import List, Optional, Dict, Any
import os

from openai import OpenAI
client = OpenAI(base_url=os.getenv("AGENT_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY", "dummy-key"))

# Additional litellm logging suppression
litellm.suppress_debug_info = True
litellm.set_verbose = False

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME
from tau_bench.agents.sanitizer import sanitize_message, coerce_content_to_str

import logging
logging.basicConfig(level=logging.INFO)

## suppress litellm logging
logging.getLogger('litellm').setLevel(logging.WARNING)
logging.getLogger('LiteLLM').setLevel(logging.WARNING)
logging.getLogger('LiteLLM Proxy').setLevel(logging.WARNING)
logging.getLogger('LiteLLM Router').setLevel(logging.WARNING)
logging.getLogger('litellm.llms').setLevel(logging.WARNING)
logging.getLogger('litellm.llms.huggingface').setLevel(logging.WARNING)
logging.getLogger('litellm.llms.huggingface.chat').setLevel(logging.WARNING)
logging.getLogger('litellm.llms.huggingface.chat.transformation').setLevel(logging.WARNING)


def get_conversation_context(messages: List[Dict[str, Any]]) -> str:
    context = ""
    for message in messages:
        context += f"{message['role']}: {message['content']}\n"
        if "tool_calls" in message:
            tool_call = message["tool_calls"][0]
            context+= f"Tool Call : {json.dumps(tool_call)}"
    return context


class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        
        
    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30, mode: str = "train", budget: int = 4
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        token_info = []
        trajectory_data = []  # Collect trajectory data for each step
        ground_truth_actions = env.task.actions
        ground_truth_actions = [action.model_dump() for action in ground_truth_actions]
        for _ in range(max_num_steps):
            # res = completion(
            #     messages=messages,
            #     model=self.model,
            #     custom_llm_provider=self.provider,
            #     tools=self.tools_info,
            #     temperature=self.temperature,
            #     # budget=budget,
            #     # return_response_only=False,
            #     api_base=os.getenv("AGENT_BASE_URL"),
            #     api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
            #     drop_params=True,
            #     # extra_body={
            #     #     'extra_data': {'ground_truth': ground_truth_actions}
            #     # }
            # )
            # print(res)
            res = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools_info,
                temperature=self.temperature,
                extra_body={
                    'return_response_only': False,
                    'budget': budget,
                }
            )
            next_message = res.choices[0].message.model_dump()
            # next_message = res.choices[0].message.provider_specific_fields['choices'][0]['message']
            
            if next_message["role"] == "assistant":
                if 'content' not in next_message:
                    next_message["content"] = ""
                if next_message["content"] is None:
                    next_message["content"] = ""
            # Normalize assistant message content
            if next_message.get("role") == "assistant":
                # coerce None/list -> string
                next_message["content"] = coerce_content_to_str(next_message.get("content"))
                # If it decided to call tools, keep only the first (your policy)
                if next_message.get("tool_calls"):
                    next_message["tool_calls"] = next_message["tool_calls"][:1]
            token_info.append(res.usage.model_dump())
            # total_cost += res._hidden_params["response_cost"] if hasattr(res, "_hidden_params") else 0
            total_cost+=0
            action = message_to_action(next_message)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            # Collect trajectory data for this step
            step_data = {
                "conversation_context": get_conversation_context(messages),
                "selected_message": res.metadata['responses'][res.metadata['selected_index']],
                "all_candidates": res.metadata['responses'],
                "scores": res.metadata['scores'],
                
            }
            trajectory_data.append(step_data)

            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(

                    [
                        sanitize_message(next_message),
                        sanitize_message({
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": str(env_response.observation),
                        }),
                    ]
                )
            else:
                messages.extend(
                    [
                        sanitize_message(next_message),
                        sanitize_message({"role": "user", "content": str(env_response.observation)}),
                    ]
                )
            if env_response.done:
                break
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
            token_info=token_info,
            trajectory_data=trajectory_data,
        )


def message_to_action(
    message: Dict[str, Any],
) -> Action:
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})
    