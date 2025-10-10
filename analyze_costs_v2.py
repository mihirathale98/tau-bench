#!/usr/bin/env python3
"""
Analyze token usage and costs from tau-bench results with per-trial breakdown support.

This version (v2) supports the new trial-based token_info structure:
  {
    "metadata": {...},
    "trials": {
      "0": [...],
      "1": [...],
      ...
    }
  }

Features:
- Per-trial cost breakdown
- Average cost per trial
- Extra usage (judge) cost tracking with configurable model
- Backward compatible with old format (single trial)

For the simpler version without per-trial breakdown, use analyze_costs.py
"""

import json
import os
import glob
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TokenStats:
    """Token usage statistics for a run."""
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    api_calls: int
    estimated_cost: float
    extra_usage_tokens: int = 0
    extra_usage_cost: float = 0.0


# OpenAI pricing (per 1M tokens)
PRICING = {
    'gpt-5': {
        'input': 1.25,      # $1.25 per 1M tokens
        'cached': 0.125,    # $0.125 per 1M tokens
        'output': 10.00     # $10.00 per 1M tokens
    },
    'gpt-5-mini': {
        'input': 0.25,      # $0.25 per 1M tokens
        'cached': 0.025,    # $0.025 per 1M tokens
        'output': 2.00      # $2.00 per 1M tokens
    },
    'gpt-5-nano': {
        'input': 0.05,      # $0.05 per 1M tokens
        'cached': 0.005,    # $0.005 per 1M tokens
        'output': 0.40      # $0.40 per 1M tokens
    },
    'gpt-5-chat-latest': {
        'input': 1.25,      # $1.25 per 1M tokens
        'cached': 0.125,    # $0.125 per 1M tokens
        'output': 10.00     # $10.00 per 1M tokens
    },
    'gpt-5-codex': {
        'input': 1.25,      # $1.25 per 1M tokens
        'cached': 0.125,    # $0.125 per 1M tokens
        'output': 10.00     # $10.00 per 1M tokens
    },
    'gpt-4.1': {
        'input': 2.00,      # $2.00 per 1M tokens
        'cached': 0.50,     # $0.50 per 1M tokens
        'output': 8.00      # $8.00 per 1M tokens
    },
    'gpt-4.1-mini': {
        'input': 0.40,      # $0.40 per 1M tokens
        'cached': 0.10,     # $0.10 per 1M tokens
        'output': 1.60      # $1.60 per 1M tokens
    },
    'gpt-4.1-nano': {
        'input': 0.10,      # $0.10 per 1M tokens
        'cached': 0.025,    # $0.025 per 1M tokens
        'output': 0.40      # $0.40 per 1M tokens
    },
    'gpt-4o': {
        'input': 2.50,      # $2.50 per 1M tokens
        'cached': 1.25,     # $1.25 per 1M tokens
        'output': 10.00     # $10.00 per 1M tokens
    }
}

# Model to use for extra_usage token cost calculations
EXTRA_USAGE_MODEL = 'gpt-4.1-mini'


def extract_model_from_filename(filename: str) -> str:
    """Extract model name from filename."""
    if 'gpt-5-chat-latest' in filename:
        return 'gpt-5-chat-latest'
    elif 'gpt-5-codex' in filename:
        return 'gpt-5-codex'
    elif 'gpt-5-mini' in filename:
        return 'gpt-5-mini'
    elif 'gpt-5-nano' in filename:
        return 'gpt-5-nano'
    elif 'gpt-5' in filename:
        return 'gpt-5'
    elif 'gpt-4.1-mini' in filename:
        return 'gpt-4.1-mini'
    elif 'gpt-4.1-nano' in filename:
        return 'gpt-4.1-nano'
    elif 'gpt-4.1' in filename:
        return 'gpt-4.1'
    elif 'gpt-4o' in filename:
        return 'gpt-4o'
    else:
        return 'unknown'


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
    """Calculate cost based on token usage and model pricing."""
    if model not in PRICING:
        print(f"Warning: Unknown model '{model}', using gpt-4o pricing")
        model = 'gpt-4o'

    pricing = PRICING[model]

    # Calculate input cost (subtract cached tokens from regular input)
    regular_input_tokens = max(0, prompt_tokens - cached_tokens)
    input_cost = (regular_input_tokens * pricing['input'] + cached_tokens * pricing['cached']) / 1000000

    # Calculate output cost
    output_cost = completion_tokens * pricing['output'] / 1000000

    return input_cost + output_cost


def analyze_task_list(task_list: List[Dict], model: str) -> TokenStats:
    """Analyze a list of tasks and calculate token statistics."""
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    api_calls = 0
    extra_usage_cost = 0.0
    extra_usage_tokens = 0
    total_cost = 0.0

    # Handle nested structure where each task has multiple usage entries
    for item in task_list:
        if isinstance(item, dict) and 'usage' in item:
            # New format with task_id and usage array
            for call in item['usage']:
                api_calls += 1
                # Sum up token usage
                total_tokens += call.get('total_tokens', 0)
                prompt_tokens += call.get('prompt_tokens', 0)
                completion_tokens += call.get('completion_tokens', 0)

                # Get cached tokens if available
                prompt_details = call.get('prompt_tokens_details', {})
                call_cached = prompt_details.get('cached_tokens', 0) if prompt_details else 0
                cached_tokens += call_cached

                # Calculate cost for this API call
                call_cost = calculate_cost(
                    model,
                    call.get('prompt_tokens', 0),
                    call.get('completion_tokens', 0),
                    call_cached
                )
                total_cost += call_cost

                # Handle extra_usage if present
                if 'extra_usage' in call:
                    extra = call['extra_usage']
                    extra_prompt = extra.get('prompt_tokens', 0)
                    extra_completion = extra.get('completion_tokens', 0)
                    extra_total = extra.get('total_tokens', 0)

                    # For extra_usage, we don't have cached token info, so pass 0
                    extra_call_cost = calculate_cost(
                        EXTRA_USAGE_MODEL,
                        extra_prompt,
                        extra_completion,
                        0  # No cached tokens in extra_usage
                    )
                    extra_usage_cost += extra_call_cost
                    extra_usage_tokens += extra_total
        else:
            # Old format - direct list of usage objects
            call = item
            api_calls += 1
            # Sum up token usage
            total_tokens += call.get('total_tokens', 0)
            prompt_tokens += call.get('prompt_tokens', 0)
            completion_tokens += call.get('completion_tokens', 0)

            # Get cached tokens if available
            prompt_details = call.get('prompt_tokens_details', {})
            call_cached = prompt_details.get('cached_tokens', 0) if prompt_details else 0
            cached_tokens += call_cached

            # Calculate cost for this API call
            call_cost = calculate_cost(
                model,
                call.get('prompt_tokens', 0),
                call.get('completion_tokens', 0),
                call_cached
            )
            total_cost += call_cost

            # Handle extra_usage if present
            if 'extra_usage' in call:
                extra = call['extra_usage']
                extra_prompt = extra.get('prompt_tokens', 0)
                extra_completion = extra.get('completion_tokens', 0)
                extra_total = extra.get('total_tokens', 0)

                # For extra_usage, we don't have cached token info, so pass 0
                extra_call_cost = calculate_cost(
                    EXTRA_USAGE_MODEL,
                    extra_prompt,
                    extra_completion,
                    0  # No cached tokens in extra_usage
                )
                extra_usage_cost += extra_call_cost
                extra_usage_tokens += extra_total

    # Add extra_usage_cost to total_cost
    total_cost += extra_usage_cost

    return TokenStats(
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        api_calls=api_calls,
        estimated_cost=total_cost,
        extra_usage_tokens=extra_usage_tokens,
        extra_usage_cost=extra_usage_cost
    )


def analyze_token_file(filepath: str) -> Tuple[Dict[str, TokenStats], TokenStats, str]:
    """Analyze a single token info JSON file.

    Returns:
        - per_trial_stats: Dict mapping trial number to TokenStats
        - total_stats: TokenStats for all trials combined
        - model: The model name
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Extract model from metadata if available, otherwise from filename
    model = 'unknown'
    if 'metadata' in data and 'model' in data['metadata']:
        model = data['metadata']['model']
    else:
        model = extract_model_from_filename(os.path.basename(filepath))

    # Get token usage data - handle both old and new format
    per_trial_stats = {}

    if 'trials' in data:
        # New format with trials as high-level key
        for trial_key, trial_tasks in data['trials'].items():
            per_trial_stats[trial_key] = analyze_task_list(trial_tasks, model)
    elif 'token_info' in data:
        # Old format - treat as single trial "0"
        per_trial_stats['0'] = analyze_task_list(data['token_info'], model)
    else:
        # Very old format
        if isinstance(data, list):
            per_trial_stats['0'] = analyze_task_list(data, model)
        else:
            print(f"Warning: Unexpected data format in {filepath}")
            return {}, TokenStats(0, 0, 0, 0, 0, 0.0), model

    # Calculate total stats across all trials
    total_stats = TokenStats(
        total_tokens=sum(s.total_tokens for s in per_trial_stats.values()),
        prompt_tokens=sum(s.prompt_tokens for s in per_trial_stats.values()),
        completion_tokens=sum(s.completion_tokens for s in per_trial_stats.values()),
        cached_tokens=sum(s.cached_tokens for s in per_trial_stats.values()),
        api_calls=sum(s.api_calls for s in per_trial_stats.values()),
        estimated_cost=sum(s.estimated_cost for s in per_trial_stats.values()),
        extra_usage_tokens=sum(s.extra_usage_tokens for s in per_trial_stats.values()),
        extra_usage_cost=sum(s.extra_usage_cost for s in per_trial_stats.values())
    )

    return per_trial_stats, total_stats, model


def save_results_to_file(results: List[Tuple[str, Dict[str, TokenStats], TokenStats, str]], output_file: str, total_cost: float, total_extra_cost: float, total_extra_tokens: int):
    """Save analysis results to a JSON file."""
    output_data = {
        "summary": {
            "total_runs": len(results),
            "total_estimated_cost": total_cost,
            "main_agent_cost": total_cost - total_extra_cost,
            "extra_usage_cost": total_extra_cost,
            "extra_usage_tokens": total_extra_tokens,
            "extra_usage_model": EXTRA_USAGE_MODEL,
            "analysis_timestamp": json.dumps(None)  # Will be filled by json.dumps default
        },
        "runs": [],
        "model_summary": {}
    }

    # Add individual run data
    for run_name, per_trial_stats, total_stats, model in results:
        run_data = {
            "run_name": run_name,
            "model": model,
            "total": {
                "api_calls": total_stats.api_calls,
                "total_tokens": total_stats.total_tokens,
                "prompt_tokens": total_stats.prompt_tokens,
                "completion_tokens": total_stats.completion_tokens,
                "cached_tokens": total_stats.cached_tokens,
                "estimated_cost": total_stats.estimated_cost,
                "extra_usage_tokens": total_stats.extra_usage_tokens,
                "extra_usage_cost": total_stats.extra_usage_cost,
                "extra_usage_model": EXTRA_USAGE_MODEL
            },
            "trials": {}
        }

        # Add per-trial data
        for trial_key, trial_stats in per_trial_stats.items():
            run_data["trials"][trial_key] = {
                "api_calls": trial_stats.api_calls,
                "total_tokens": trial_stats.total_tokens,
                "prompt_tokens": trial_stats.prompt_tokens,
                "completion_tokens": trial_stats.completion_tokens,
                "cached_tokens": trial_stats.cached_tokens,
                "estimated_cost": trial_stats.estimated_cost,
                "extra_usage_tokens": trial_stats.extra_usage_tokens,
                "extra_usage_cost": trial_stats.extra_usage_cost
            }

        # Add average per trial
        num_trials = len(per_trial_stats)
        if num_trials > 0:
            run_data["average_per_trial"] = {
                "api_calls": total_stats.api_calls / num_trials,
                "total_tokens": total_stats.total_tokens / num_trials,
                "prompt_tokens": total_stats.prompt_tokens / num_trials,
                "completion_tokens": total_stats.completion_tokens / num_trials,
                "cached_tokens": total_stats.cached_tokens / num_trials,
                "estimated_cost": total_stats.estimated_cost / num_trials,
                "extra_usage_tokens": total_stats.extra_usage_tokens / num_trials,
                "extra_usage_cost": total_stats.extra_usage_cost / num_trials
            }

        output_data["runs"].append(run_data)
    
    # Group by model for summary
    model_costs = {}
    total_extra_cost_calc = 0.0
    total_extra_tokens_calc = 0
    for _, per_trial_stats, total_stats, model in results:
        if model not in model_costs:
            model_costs[model] = {'cost': 0, 'runs': 0, 'tokens': 0}
        # Only add the main agent cost (exclude extra_usage_cost)
        main_agent_cost = total_stats.estimated_cost - total_stats.extra_usage_cost
        model_costs[model]['cost'] += main_agent_cost
        model_costs[model]['runs'] += 1
        model_costs[model]['tokens'] += total_stats.total_tokens
        total_extra_cost_calc += total_stats.extra_usage_cost
        total_extra_tokens_calc += total_stats.extra_usage_tokens

    # Add model summary
    for model, data in model_costs.items():
        avg_cost = data['cost'] / data['runs'] if data['runs'] > 0 else 0
        output_data["model_summary"][model] = {
            "total_cost": data['cost'],
            "average_cost": avg_cost,
            "runs": data['runs'],
            "total_tokens": data['tokens']
        }

    # Add extra usage summary
    if total_extra_cost > 0:
        output_data["extra_usage_summary"] = {
            "model": EXTRA_USAGE_MODEL,
            "total_tokens": total_extra_tokens,
            "total_cost": total_extra_cost,
            "runs": sum(1 for _, _, total_stats, _ in results if total_stats.extra_usage_cost > 0)
        }
    else:
        output_data["extra_usage_summary"] = None
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {output_file}")


def main():
    """Main function to analyze all token info files."""
    parser = argparse.ArgumentParser(description='Analyze token usage and costs from tau-bench results')
    parser.add_argument('input_dir', help='Directory containing *_token_info.json files')
    parser.add_argument('-o', '--output', help='Output JSON file to save results')
    
    args = parser.parse_args()
    
    results_dir = args.input_dir

    # Find all token info files
    token_files = glob.glob(os.path.join(results_dir, "*_token_info.json"))

    if not token_files:
        print(f"No token info files found in {results_dir}")
        return

    print("τ-bench Cost Analysis")
    print("=" * 60)
    print(f"Analyzing {len(token_files)} files from: {results_dir}")
    print()

    total_cost = 0.0
    results = []

    for filepath in sorted(token_files):
        filename = os.path.basename(filepath)
        run_name = filename.replace('_token_info.json', '')

        per_trial_stats, total_stats, model = analyze_token_file(filepath)
        results.append((run_name, per_trial_stats, total_stats, model))
        total_cost += total_stats.estimated_cost

        print(f"Run: {run_name}")
        print(f"Model: {model}")
        print(f"Number of Trials: {len(per_trial_stats)}")

        # Display per-trial costs
        if len(per_trial_stats) > 1:
            print(f"\nPer-Trial Breakdown:")
            for trial_key in sorted(per_trial_stats.keys(), key=lambda x: int(x)):
                trial_stats = per_trial_stats[trial_key]
                trial_main_cost = trial_stats.estimated_cost - trial_stats.extra_usage_cost
                print(f"  Trial {trial_key}:")
                print(f"    Main Agent ({model}): ${trial_main_cost:.6f}")
                if trial_stats.extra_usage_tokens > 0:
                    print(f"    Extra Usage ({EXTRA_USAGE_MODEL}): ${trial_stats.extra_usage_cost:.6f} ({trial_stats.extra_usage_tokens:,} tokens)")
                print(f"    TOTAL: ${trial_stats.estimated_cost:.6f} ({trial_stats.api_calls} calls, {trial_stats.total_tokens:,} tokens)")

            # Average per trial
            avg_cost = total_stats.estimated_cost / len(per_trial_stats)
            avg_main_cost = (total_stats.estimated_cost - total_stats.extra_usage_cost) / len(per_trial_stats)
            avg_extra_cost = total_stats.extra_usage_cost / len(per_trial_stats)
            print(f"\n  Average per trial:")
            print(f"    Main Agent: ${avg_main_cost:.6f}")
            if total_stats.extra_usage_cost > 0:
                print(f"    Extra Usage: ${avg_extra_cost:.6f}")
            print(f"    TOTAL: ${avg_cost:.6f}")
            print()

        # Total across all trials
        print(f"Total API Calls: {total_stats.api_calls:,}")
        print(f"Total Tokens: {total_stats.total_tokens:,}")
        print(f"  - Prompt: {total_stats.prompt_tokens:,}")
        print(f"  - Completion: {total_stats.completion_tokens:,}")
        print(f"  - Cached: {total_stats.cached_tokens:,}")

        # Cost breakdown
        main_agent_cost = total_stats.estimated_cost - total_stats.extra_usage_cost
        print(f"\nCost Breakdown:")
        print(f"  Main Agent ({model}): ${main_agent_cost:.6f}")
        if total_stats.extra_usage_tokens > 0:
            print(f"  Extra Usage ({EXTRA_USAGE_MODEL}): ${total_stats.extra_usage_cost:.6f} ({total_stats.extra_usage_tokens:,} tokens)")
        print(f"  TOTAL: ${total_stats.estimated_cost:.6f}")
        print("-" * 60)

    # Summary
    print("\nSUMMARY")
    print("=" * 60)
    print(f"Total Runs: {len(results)}")
    print(f"Total Estimated Cost: ${total_cost:.6f}")
    print()

    # Sort by cost for top spenders
    results_by_cost = sorted(results, key=lambda x: x[2].estimated_cost, reverse=True)
    print("Most Expensive Runs:")
    for i, (name, per_trial_stats, total_stats, model) in enumerate(results_by_cost[:5], 1):
        print(f"{i}. {name} ({model}): ${total_stats.estimated_cost:.6f}")

    # Group by model
    print("\nCost by Model:")
    model_costs = {}
    extra_model_costs = {EXTRA_USAGE_MODEL: {'cost': 0.0, 'runs': 0, 'tokens': 0}}
    total_extra_cost = 0.0
    total_extra_tokens = 0
    for _, per_trial_stats, total_stats, model in results:
        if model not in model_costs:
            model_costs[model] = {'cost': 0, 'runs': 0, 'tokens': 0}
        # Only add the main agent cost (exclude extra_usage_cost)
        main_agent_cost = total_stats.estimated_cost - total_stats.extra_usage_cost
        model_costs[model]['cost'] += main_agent_cost
        model_costs[model]['runs'] += 1
        model_costs[model]['tokens'] += total_stats.total_tokens

        # Accumulate extra usage costs separately
        total_extra_cost += total_stats.extra_usage_cost
        total_extra_tokens += total_stats.extra_usage_tokens
        if total_stats.extra_usage_cost > 0:
            extra_model_costs[EXTRA_USAGE_MODEL]['cost'] += total_stats.extra_usage_cost
            extra_model_costs[EXTRA_USAGE_MODEL]['runs'] += 1
            extra_model_costs[EXTRA_USAGE_MODEL]['tokens'] += total_stats.extra_usage_tokens

    # Combine model costs with extra model costs for display
    all_model_costs = {**model_costs, **extra_model_costs}

    for model, data in sorted(all_model_costs.items(), key=lambda x: x[1]['cost'], reverse=True):
        if data['cost'] > 0:  # Only show models with actual costs
            avg_cost = data['cost'] / data['runs'] if data['runs'] > 0 else 0
            print(f"{model}: ${data['cost']:.6f} total, ${avg_cost:.6f} avg ({data['runs']} runs, {data['tokens']:,} tokens)")

    # Display extra usage costs
    if total_extra_tokens > 0:
        print(f"\nExtra Usage ({EXTRA_USAGE_MODEL}):")
        print(f"  Total Tokens: {total_extra_tokens:,}")
        print(f"  Total Cost: ${total_extra_cost:.6f}")

    # Final cost breakdown
    print("\n" + "=" * 60)
    print("FINAL COST BREAKDOWN")
    print("=" * 60)
    main_cost = total_cost - total_extra_cost
    print(f"Main Agent Cost: ${main_cost:.6f}")
    if total_extra_cost > 0:
        print(f"Extra Usage Cost ({EXTRA_USAGE_MODEL}): ${total_extra_cost:.6f}")
    print(f"TOTAL COST: ${total_cost:.6f}")
    print("=" * 60)

    # Save to file if requested
    if args.output:
        save_results_to_file(results, args.output, total_cost, total_extra_cost, total_extra_tokens)


if __name__ == "__main__":
    main()