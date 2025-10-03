#!/usr/bin/env python3
"""
Analyze token usage and costs from tau-bench airline OpenAI results.
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


def analyze_token_file(filepath: str) -> Tuple[TokenStats, str]:
    """Analyze a single token info JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Extract model from metadata if available, otherwise from filename
    model = 'unknown'
    if 'metadata' in data and 'model' in data['metadata']:
        model = data['metadata']['model']
    else:
        model = extract_model_from_filename(os.path.basename(filepath))

    # Get token usage data - handle both old and new format
    token_data = data.get('token_info', data) if 'token_info' in data else data
    if not isinstance(token_data, list):
        print(f"Warning: Unexpected data format in {filepath}")
        return TokenStats(0, 0, 0, 0, 0, 0.0), model

    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    api_calls = 0

    total_cost = 0.0

    # Handle nested structure where each task has multiple usage entries
    for item in token_data:
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

    return TokenStats(
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        api_calls=api_calls,
        estimated_cost=total_cost
    ), model


def save_results_to_file(results: List[Tuple[str, TokenStats, str]], output_file: str, total_cost: float):
    """Save analysis results to a JSON file."""
    output_data = {
        "summary": {
            "total_runs": len(results),
            "total_estimated_cost": total_cost,
            "analysis_timestamp": json.dumps(None)  # Will be filled by json.dumps default
        },
        "runs": [],
        "model_summary": {}
    }
    
    # Add individual run data
    for run_name, stats, model in results:
        output_data["runs"].append({
            "run_name": run_name,
            "model": model,
            "api_calls": stats.api_calls,
            "total_tokens": stats.total_tokens,
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens,
            "cached_tokens": stats.cached_tokens,
            "estimated_cost": stats.estimated_cost
        })
    
    # Group by model for summary
    model_costs = {}
    for _, stats, model in results:
        if model not in model_costs:
            model_costs[model] = {'cost': 0, 'runs': 0, 'tokens': 0}
        model_costs[model]['cost'] += stats.estimated_cost
        model_costs[model]['runs'] += 1
        model_costs[model]['tokens'] += stats.total_tokens
    
    # Add model summary
    for model, data in model_costs.items():
        avg_cost = data['cost'] / data['runs'] if data['runs'] > 0 else 0
        output_data["model_summary"][model] = {
            "total_cost": data['cost'],
            "average_cost": avg_cost,
            "runs": data['runs'],
            "total_tokens": data['tokens']
        }
    
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

        stats, model = analyze_token_file(filepath)
        results.append((run_name, stats, model))
        total_cost += stats.estimated_cost

        print(f"Run: {run_name}")
        print(f"Model: {model}")
        print(f"API Calls: {stats.api_calls:,}")
        print(f"Total Tokens: {stats.total_tokens:,}")
        print(f"  - Prompt: {stats.prompt_tokens:,}")
        print(f"  - Completion: {stats.completion_tokens:,}")
        print(f"  - Cached: {stats.cached_tokens:,}")
        print(f"Estimated Cost: ${stats.estimated_cost:.6f}")
        print("-" * 60)

    # Summary
    print("\nSUMMARY")
    print("=" * 60)
    print(f"Total Runs: {len(results)}")
    print(f"Total Estimated Cost: ${total_cost:.6f}")
    print()

    # Sort by cost for top spenders
    results_by_cost = sorted(results, key=lambda x: x[1].estimated_cost, reverse=True)
    print("Most Expensive Runs:")
    for i, (name, stats, model) in enumerate(results_by_cost[:5], 1):
        print(f"{i}. {model}: ${stats.estimated_cost:.6f}")

    # Group by model
    print("\nCost by Model:")
    model_costs = {}
    for _, stats, model in results:
        if model not in model_costs:
            model_costs[model] = {'cost': 0, 'runs': 0, 'tokens': 0}
        model_costs[model]['cost'] += stats.estimated_cost
        model_costs[model]['runs'] += 1
        model_costs[model]['tokens'] += stats.total_tokens

    for model, data in sorted(model_costs.items(), key=lambda x: x[1]['cost'], reverse=True):
        avg_cost = data['cost'] / data['runs'] if data['runs'] > 0 else 0
        print(f"{model}: ${data['cost']:.6f} total, ${avg_cost:.6f} avg ({data['runs']} runs, {data['tokens']:,} tokens)")

    # Save to file if requested
    if args.output:
        save_results_to_file(results, args.output, total_cost)


if __name__ == "__main__":
    main()