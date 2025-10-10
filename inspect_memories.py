#!/usr/bin/env python3
"""
Script to inspect memories stored in mem0 for tau-bench experiments.

Usage:
    python inspect_memories.py --env retail
    python inspect_memories.py --env retail --query "cancel order"
    python inspect_memories.py --env retail --stats
"""

import argparse
from tau_bench.agents.mem0_module import Mem0Module
import json


def main():
    parser = argparse.ArgumentParser(description="Inspect mem0 memories")
    parser.add_argument("--env", required=True, choices=["retail", "airline"],
                        help="Environment (retail or airline)")
    parser.add_argument("--query", type=str, help="Search query to retrieve similar memories")
    parser.add_argument("--limit", type=int, default=10, help="Number of memories to retrieve")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    parser.add_argument("--all", action="store_true", help="Show all memories (no search)")
    parser.add_argument("--min-reward", type=float, help="Filter by minimum reward")

    args = parser.parse_args()

    # Initialize memory module (don't delete existing memories!)
    collection_name = f"memory_{args.env}"
    print(f"Loading memories from collection: {collection_name}\n")
    memory = Mem0Module(collection_name=collection_name, delete_existing=False)

    # Show statistics
    stats = memory.get_stats()
    print(f"📊 Statistics:")
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   Collection: {stats['collection_name']}\n")

    if args.stats:
        return

    # Retrieve memories
    if args.all:
        print("📚 Retrieving ALL memories...\n")
        all_memories = memory.memory.get_all(user_id=collection_name)

        for i, mem in enumerate(all_memories, 1):
            metadata = mem.get('metadata', {})
            reward = metadata.get('reward', 0.0)

            # Apply reward filter if specified
            if args.min_reward and reward < args.min_reward:
                continue

            print(f"{'='*80}")
            print(f"Memory {i}")
            print(f"{'='*80}")
            print(f"Task Index: {metadata.get('task_index', 'N/A')}")
            print(f"Reward: {reward:.2f}")
            print(f"Intent: {metadata.get('intent', 'N/A')}")
            print(f"\nMemory Content:")
            print(f"{'-'*80}")
            print(mem.get('memory', 'N/A'))
            print(f"\n")

    elif args.query:
        print(f"🔍 Searching for: '{args.query}'\n")

        filter_dict = None
        if args.min_reward:
            filter_dict = {'reward': args.min_reward}

        memories = memory.retrieve_memory(
            intent=args.query,
            filter=filter_dict,
            limit=args.limit
        )

        print(f"Found {len(memories)} relevant memories:\n")

        for i, mem in enumerate(memories, 1):
            print(f"{'='*80}")
            print(f"Memory {i}")
            print(f"{'='*80}")
            print(f"Task Index: {mem['task_index']}")
            print(f"Reward: {mem['reward']:.2f}")
            print(f"Intent: {mem['intent']}")
            print(f"\nMemory Content:")
            print(f"{'-'*80}")
            print(mem['memory'])
            print(f"\n")

    else:
        print("ℹ️  Use --query to search, --all to show all memories, or --stats for statistics only")


if __name__ == "__main__":
    main()
