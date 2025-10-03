# Copyright Sierra

import json
from litellm import completion
import litellm
from typing import List, Optional, Dict, Any

# Additional litellm logging suppression
litellm.suppress_debug_info = True
litellm.set_verbose = False

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME

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



## custom memory logger - only logs memory generation
memory_logger = logging.getLogger('memory_generation')
memory_logger.setLevel(logging.INFO)
memory_handler = logging.FileHandler('memory_generation.log')
memory_formatter = logging.Formatter('%(asctime)s - MEMORY - %(message)s')
memory_handler.setFormatter(memory_formatter)
memory_logger.addHandler(memory_handler)
# Prevent memory logs from propagating to parent loggers
memory_logger.propagate = False


class MemoryAgent(Agent):
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
        
    
        
    def get_instruction_summary(self, instruction: str) -> str:
        
        prompt = f"""
        Briefly summarize the following user instruction as intent. 
        It will later be used to retrive summary of relevant conversation from memory. 
        Do not add any specifics(personal information, order id, etc.) to the summary. 
        Keep it general.
        
        
        Example: 
        Intent: Cancel a pending order due to a better price found elsewhere.
        
        Only output the intent, no other text.
        User instruction: {instruction}
        Intent:
        """
        res = completion(
            model='openai/gpt-4o-mini',
            custom_llm_provider='openai',
            messages=[{"role": "user", "content": prompt}],
        )
        return res.choices[0].message.content
    
    def retrieve_memory(self, memory: Any, instruction_summary: str, limit: int = 4) -> str:
        memories = memory.retrieve_memory(instruction_summary, limit=limit)
        memories_str = ""
        if len(memories) > 0:
            memories_str = "\n".join([entry['memory'] for entry in memories])
        
        return memories_str
        
        
    def generate_traj_summary(self, messages: List[Dict[str, Any]]) -> str:
        system_prompt = """
        You are a procedural memory writer. Your task is to distill past user-agent trajectories into short, reusable procedural memory cards for an agent to follow.

        Output in plain text (not JSON).
        Sections: User Intent, Steps-tool mapping, Learning, Correction needed, Deviation.
        Be concise (<250 tokens), generalizable, and accurate. Do not deviate from the actual policies.
        Mask sensitive details (e.g., card last4 only).
        Do not include long prose or story-specific details.
        """


        instruction = """
        Given the following trajectory data, generate a concise procedural memory card.

        trajectory:
        {sample_traj_messages}

        fixed policies:
        {fixed_policies}

        Requirements:
        Start with Intent (1 line).
        A map of steps to tools, highlighting what tool was used for what action domain.
        What were overall learning from the trajectory? (1-2 lines)
        What are the correction needed? (1-2 lines)
        Note if trajectory deviated from the fixed policies.(write as "should not have done....")
        Keep length <250 tokens.
        Only output the procedural memory card, nothing else.
        """
        messages_str = json.dumps(messages)
        instruction = instruction.format(sample_traj_messages=messages_str, fixed_policies=self.wiki)

        res = completion(
            model="openai/gpt-4o",
            custom_llm_provider="openai",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ],
            temperature=0.6,
            max_tokens=1024,
        )
        return res.choices[0].message.content


        

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30, memory: Optional[Any] = None, mode: str = "train"
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        task_instruction = env.task.instruction
        instruction_summary = self.get_instruction_summary(task_instruction)
        
        
        wiki = self.wiki
        if mode == "test":
            retrieved_memory = self.retrieve_memory(memory, instruction_summary, limit=4)
        
            if len(retrieved_memory) > 0:
                additional_prompt = f"""
                You have also been provided with memory recorded from past trajectories, which can be used to guide your actions.
                
                Past experiences:
                {retrieved_memory}
                """
                wiki = wiki + "\n\n" + additional_prompt
        
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": wiki},
            {"role": "user", "content": obs},
        ]
        structured_memory = None
        for _ in range(max_num_steps):
            res = completion(
                messages=messages,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=self.tools_info,
                temperature=self.temperature,
            )
            next_message = res.choices[0].message.model_dump()
            total_cost += res._hidden_params["response_cost"]
            action = message_to_action(next_message)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(

                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
                
            if env_response.done:
                if mode == "train":
                    traj_summary = self.generate_traj_summary(messages)
                    structured_memory = {"task_index": task_index, "reward": reward, "memory": traj_summary, "intent": instruction_summary}
                    memory.add_memory(structured_memory)
                break
        if structured_memory is None:
            if mode == "train":
                traj_summary = self.generate_traj_summary(messages)
                structured_memory = {"task_index": task_index, "reward": reward, "memory": traj_summary, "intent": instruction_summary}
                memory.add_memory(structured_memory)
            
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
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