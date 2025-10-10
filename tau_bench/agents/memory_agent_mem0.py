# Copyright Sierra

"""
Memory-augmented agent using mem0 for tau-bench.

This agent creates procedural memories from training trajectories and retrieves
them during test time to guide decision-making. It uses mem0 for memory storage
and retrieval.

Key improvements over the original memory_agent.py:
- Uses mem0 instead of manual Qdrant management
- Actually logs generated memories to memory_generation_mem0.log
- Filters memories by reward threshold (only stores reward >= 0.8)
- Retrieves only high-quality memories during test time
"""

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

# Suppress litellm logging
logging.getLogger('litellm').setLevel(logging.WARNING)
logging.getLogger('LiteLLM').setLevel(logging.WARNING)
logging.getLogger('LiteLLM Proxy').setLevel(logging.WARNING)
logging.getLogger('LiteLLM Router').setLevel(logging.WARNING)
logging.getLogger('litellm.llms').setLevel(logging.WARNING)
logging.getLogger('litellm.llms.huggingface').setLevel(logging.WARNING)
logging.getLogger('litellm.llms.huggingface.chat').setLevel(logging.WARNING)
logging.getLogger('litellm.llms.huggingface.chat.transformation').setLevel(logging.WARNING)

# Custom memory logger - logs all generated memories
memory_logger = logging.getLogger('memory_generation_mem0')
memory_logger.setLevel(logging.INFO)
memory_handler = logging.FileHandler('memory_generation_mem0.log')
memory_formatter = logging.Formatter('%(asctime)s - MEMORY - %(message)s')
memory_handler.setFormatter(memory_formatter)
memory_logger.addHandler(memory_handler)
# Prevent memory logs from propagating to parent loggers
memory_logger.propagate = False


# Configuration constants
MIN_REWARD_THRESHOLD = 0.8  # Only store/retrieve memories with reward >= 0.8
MEMORY_RETRIEVAL_LIMIT = 4  # Number of memories to retrieve during test


class MemoryAgentMem0(Agent):
    """
    Memory-augmented agent using mem0 for storage and retrieval.

    During training:
    - Runs tasks and generates procedural memory cards from trajectories
    - Only stores high-quality memories (reward >= MIN_REWARD_THRESHOLD)
    - Logs all generated memories for inspection

    During testing:
    - Retrieves relevant memories based on task intent
    - Injects retrieved memories into system prompt
    - Agent uses past experiences to guide decisions
    """

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
        """
        Summarize user instruction as a generalized intent for memory retrieval.

        This creates a high-level intent that can match similar scenarios,
        without including specific details like order IDs, names, etc.
        """
        prompt = f"""
        Briefly summarize the following user instruction as intent.
        It will later be used to retrieve summary of relevant conversation from memory.
        Do not add any specifics (personal information, order id, etc.) to the summary.
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

    def retrieve_memory(self, memory: Any, instruction_summary: str, limit: int = MEMORY_RETRIEVAL_LIMIT) -> str:
        """
        Retrieve relevant memories from mem0 based on instruction summary.

        Args:
            memory: Mem0Module instance
            instruction_summary: Generalized intent for retrieval
            limit: Number of memories to retrieve

        Returns:
            Formatted string of retrieved memories
        """
        # Filter for high-quality memories only
        memories = memory.retrieve_memory(
            instruction_summary,
            filter={'reward': MIN_REWARD_THRESHOLD},
            limit=limit
        )

        # memories_str = ""
        # if len(memories) > 0:
        #     memories_str = "\n\n---\n\n".join([entry['memory'] for entry in memories])
        #     memory_logger.info(f"Retrieved {len(memories)} memories for intent: {instruction_summary}")

        return memories

    def generate_traj_summary(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate a procedural memory card from a trajectory.

        This uses GPT-4o to distill the trajectory into a reusable memory card
        that respects the fixed policy guidelines. The memory includes:
        - User Intent
        - Steps-to-tool mapping
        - Overall learning
        - Corrections needed
        - Policy deviations (if any)
        """
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
        self,
        env: Env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
        memory: Optional[Any] = None,
        mode: str = "train",
        budget: int = 4
    ) -> SolveResult:
        """
        Solve a task with optional memory augmentation.

        Args:
            env: Environment instance
            task_index: Task ID
            max_num_steps: Maximum conversation turns
            memory: Mem0Module instance
            mode: "train" (create memories) or "test" (use memories)
            budget: Budget parameter (unused in current implementation)

        Returns:
            SolveResult with reward, trajectory, and metadata
        """
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        task_instruction = env.task.instruction
        instruction_summary = self.get_instruction_summary(task_instruction)
        agent_id = "our_agent"

        wiki = self.wiki

        # In test mode, retrieve and inject relevant memories
        if mode == "test":
            retrieved_memory = self.retrieve_memory(memory, instruction_summary, limit=MEMORY_RETRIEVAL_LIMIT)
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

        # Main conversation loop
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

            # Check if task is done
            if env_response.done:
                # In train mode, generate and store memory if reward is good enough
                if mode == "train" and memory is not None:
                    traj_summary = self.generate_traj_summary(messages)

                    # Log the generated memory
                    memory_logger.info(f"\n{'='*80}\nTask {task_index} | Reward: {reward:.2f}\n{'-'*80}\n{traj_summary}\n{'='*80}\n")

                    # Only store high-quality memories
                    if reward >= MIN_REWARD_THRESHOLD:
                        structured_memory = {
                            "task_index": task_index,
                            "reward": reward,
                            "trajectory": messages,
                            "agent_id": agent_id,
                            "intent": instruction_summary
                        }
                        print("Calling add_memory")
                        memory.add_memory(structured_memory)
                        memory_logger.info(f"✓ Stored memory for task {task_index} (reward: {reward:.2f})")
                    else:
                        memory_logger.info(f"✗ Skipped storing memory for task {task_index} (reward: {reward:.2f} < {MIN_REWARD_THRESHOLD})")

                break

        # If loop ended without done flag, still generate memory in train mode
        if structured_memory is None and mode == "train" and memory is not None:
            traj_summary = self.generate_traj_summary(messages)

            # Log the generated memory
            memory_logger.info(f"\n{'='*80}\nTask {task_index} (INCOMPLETE) | Reward: {reward:.2f}\n{'-'*80}\n{traj_summary}\n{'='*80}\n")

            # Only store high-quality memories
            if reward >= MIN_REWARD_THRESHOLD:
                structured_memory = {
                    "task_index": task_index,
                    "reward": reward,
                    "trajectory": messages,
                    "agent_id": agent_id,
                    "intent": instruction_summary
                }
                memory.add_memory(structured_memory)
                memory_logger.info(f"✓ Stored memory for task {task_index} (reward: {reward:.2f})")
            else:
                memory_logger.info(f"✗ Skipped storing memory for task {task_index} (reward: {reward:.2f} < {MIN_REWARD_THRESHOLD})")

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )


def message_to_action(message: Dict[str, Any]) -> Action:
    """Convert LLM message to Action."""
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})
