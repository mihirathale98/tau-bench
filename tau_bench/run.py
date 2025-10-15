# Copyright Sierra

import os
import json
import random
import traceback
from math import comb
import multiprocessing
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from tau_bench.envs import get_env
from tau_bench.agents.base import Agent
from tau_bench.types import EnvRunResult, RunConfig
from litellm import provider_list
from tau_bench.envs.user import UserStrategy


def run(config: RunConfig) -> List[EnvRunResult]:
    assert config.env in ["retail", "airline"], "Only retail and airline envs are supported"
    assert config.model_provider in provider_list, "Invalid model provider"
    assert config.user_model_provider in provider_list, "Invalid user model provider"
    assert config.agent_strategy in ["tool-calling", "act", "react", "few-shot", "langgraph", "memory"], "Invalid agent strategy"
    assert config.task_split in ["train", "test", "dev"], "Invalid task split"
    assert config.user_strategy in [item.value for item in UserStrategy], "Invalid user strategy"

    random.seed(config.seed)
    time_str = datetime.now().strftime("%m%d%H%M%S")
    ckpt_path = f"{config.log_dir}/{config.agent_strategy}-{config.model.split('/')[-1]}-{config.temperature}_range_{config.start_index}-{config.end_index}_user-{config.user_model}-{config.user_strategy}_{time_str}.json"
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    print(f"Loading user with strategy: {config.user_strategy}")
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=config,
    )
    end_index = (
        len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))
    )
    results: List[EnvRunResult] = []
    lock = multiprocessing.Lock()

    # Initialize token info accumulator with metadata
    all_token_info = {
        "metadata": {
            "model": config.model,
            "user_model": config.user_model,
            "env": config.env
        },
        "trials": {}
    }
    
    if config.task_split == "train":
        delete_existing = True
    else:
        delete_existing = False
    
    if config.task_ids and len(config.task_ids) > 0:
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(
            f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})"
    )
    for i in range(config.num_trials):
        if config.task_ids and len(config.task_ids) > 0:
            idxs = config.task_ids
        else:
            idxs = list(range(config.start_index, end_index))
        if config.shuffle:
            random.shuffle(idxs)

        def _run(idx: int) -> EnvRunResult:
            isolated_env = get_env(
                config.env,
                user_strategy=config.user_strategy,
                user_model=config.user_model,
                task_split=config.task_split,
                user_provider=config.user_model_provider,
                task_index=idx,
            )

            print(f"Running task {idx}")
            try:
                res = agent.solve(
                    env=isolated_env,
                    task_index=idx,
                    mode=config.task_split,
                    budget=config.budget,
                    max_num_steps=config.max_num_steps
                )
                token_info = res.token_info
                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=i,
                )
            except Exception as e:
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=i,
                )
                token_info = [] # Empty list
            print(
                "✅" if result.reward == 1 else "❌",
                f"task_id={idx}",
                # result.info,
            )
            print("-----")
            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path, "r") as f:
                        data = json.load(f)
                with open(ckpt_path, "w") as f:
                    json.dump(data + [result.model_dump()], f, indent=2)

                # Accumulate token info with trial as high-level key
                if token_info:  # Only add if there's token info
                    trial_key = str(i)
                    if trial_key not in all_token_info["trials"]:
                        all_token_info["trials"][trial_key] = []

                    all_token_info["trials"][trial_key].append({
                        "task_id": idx,
                        "usage": token_info,
                    })

                # Save accumulated token info
                with open(ckpt_path.replace(".json", "_token_info.json"), "w") as f:
                    json.dump(all_token_info, f, indent=2)
            return result

        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            res = list(executor.map(_run, idxs))
            results.extend(res)

    pass_hat_ks, pass_at_ks = display_metrics(results)
    with open(ckpt_path.replace(".json", "_pass_hat_ks.json"), "w") as f:
        json.dump(pass_hat_ks, f, indent=2)
    with open(ckpt_path.replace(".json", "_pass_at_ks.json"), "w") as f:
        json.dump(pass_at_ks, f, indent=2)
        
    with open(ckpt_path, "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)
        print(f"\n📄 Results saved to {ckpt_path}\n")
    return results
   
   # Both Phoenix and file export


def agent_factory(
    tools_info: List[Dict[str, Any]], wiki, config: RunConfig
) -> Agent:
    if config.agent_strategy == "tool-calling":
        # native tool calling
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent

        return ToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "act":
        # `act` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=False,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "react":
        # `react` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "few-shot":
        from tau_bench.agents.few_shot_agent import FewShotToolCallingAgent
        assert config.few_shot_displays_path is not None, "Few shot displays path is required for few-shot agent strategy"
        with open(config.few_shot_displays_path, "r") as f:
            few_shot_displays = [json.loads(line)["messages_display"] for line in f]

        return FewShotToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            few_shot_displays=few_shot_displays,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "langgraph":
        
        # LangGraph-based tool calling agent
        from tau_bench.agents.langgraph_tool_call_agent import LangGraphToolCallingAgent

        return LangGraphToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
            save_traces_locally=True,        # Enable local saving
            local_traces_path="./airline_traces_granite",
            
        )
    elif config.agent_strategy == "memory":
        from tau_bench.agents.memory_agent import MemoryAgent
        return MemoryAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"Unknown agent strategy: {config.agent_strategy}")


def display_metrics(results: List[EnvRunResult]) -> dict[int, float]:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r.trial for r in results]))
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    
    # Calculate pass@k (standard metric)
    pass_at_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_at_k = 0
        for c in c_per_task_id.values():
            # pass@k = 1 - (n-c choose k) / (n choose k)
            # where n is total trials, c is number of successes
            # Special case: if there are fewer than k failures, impossible to pick k failures
            if num_trials - c < k:
                sum_task_pass_at_k += 1.0
            else:
                sum_task_pass_at_k += 1 - (comb(num_trials - c, k) / comb(num_trials, k))
        pass_at_ks[k] = sum_task_pass_at_k / len(c_per_task_id)
    
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k (unbiased estimator)")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")
    print("📊 Pass@k (standard metric)")
    for k, pass_at_k in pass_at_ks.items():
        print(f"  k={k}: {pass_at_k}")
    return pass_hat_ks, pass_at_ks
