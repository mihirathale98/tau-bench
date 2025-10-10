# Copyright Sierra

"""
Mem0 wrapper module for tau-bench memory experiments.

This module provides a clean interface to mem0 (https://github.com/mem0ai/mem0)
that matches the API of the original MemModule (Qdrant-based), allowing easy
comparison between memory backends.

Key differences from MemModule:
- Uses mem0's managed vector DB (no manual Qdrant setup)
- mem0 handles embeddings, deduplication, and versioning
- Simpler configuration and setup
"""

from typing import Dict, List, Optional
import logging

try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    logging.warning("mem0ai not installed. Install with: pip install mem0ai")


logger = logging.getLogger(__name__)


class Mem0Module:
    """
    Wrapper around mem0's Memory API for tau-bench.

    Provides the same interface as MemModule (Qdrant-based) for easy swapping.
    """

    def __init__(
        self,
        collection_name: str,
        delete_existing: bool = False,
        config: Optional[Dict] = None
    ):
        """
        Initialize mem0 memory module.

        Args:
            collection_name: Name for the memory collection (e.g., "memory_retail")
            delete_existing: If True, clear all existing memories on init
            config: Optional mem0 configuration dict
        """
        if not MEM0_AVAILABLE:
            raise ImportError(
                "mem0ai is not installed. Install with: pip install mem0ai"
            )

        self.collection_name = collection_name

        # Default mem0 config - uses Qdrant managed by mem0
        default_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": collection_name,
                    "embedding_model_dims": 1536,  # OpenAI text-embedding-3-small
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small"
                }
            }
        }

        # Merge user config if provided
        if config:
            default_config.update(config)

        # Initialize mem0
        self.memory = Memory.from_config(default_config)

        # Clear existing memories if requested
        if delete_existing:
            try:
                # mem0's way to clear all memories for a user/collection
                # Note: mem0 organizes by user_id, so we use collection_name as user_id
                all_memories = self.memory.get_all(user_id=collection_name)
                if all_memories:
                    logger.info(f"Deleting {len(all_memories)} existing memories from {collection_name}")
                    for mem in all_memories:
                        self.memory.delete(mem['id'], user_id=collection_name)
            except Exception as e:
                logger.warning(f"Could not clear existing memories: {e}")

        logger.info(f"Initialized Mem0Module for collection: {collection_name}")

    def add_memory(self, structured_memory: Dict) -> None:
        """
        Add a memory to mem0.

        Args:
            structured_memory: Dict with keys:
                - memory: str, the actual memory text
                - intent: str, the user intent summary
                - reward: float, the task reward
                - task_index: int, the task ID
        """
        traj_memory = structured_memory["memory"]
        intent = structured_memory["intent"]
        reward = structured_memory["reward"]
        task_index = structured_memory["task_index"]

        # mem0 stores memories with metadata
        # We use collection_name as user_id to organize by environment
        self.memory.add(
            messages=traj_memory,  # The actual memory content
            user_id=self.collection_name,
            metadata={
                "intent": intent,
                "reward": reward,
                "task_index": task_index,
            }
        )

        logger.debug(f"Added memory for task {task_index} with reward {reward}")

    def retrieve_memory(
        self,
        intent: str,
        filter: Optional[Dict] = None,
        limit: int = 4
    ) -> List[Dict]:
        """
        Retrieve relevant memories based on intent.

        Args:
            intent: User intent to search for
            filter: Optional filter dict, e.g., {"reward": 0.8} for minimum reward
            limit: Maximum number of memories to retrieve

        Returns:
            List of memory dicts with keys: memory, intent, reward, task_index
        """
        try:
            # Search memories using mem0
            results = self.memory.search(
                query=intent,
                user_id=self.collection_name,
                limit=limit
            )

            # Convert mem0 format to our expected format
            memories = []
            for result in results:
                # Handle both dict and string formats from mem0
                if isinstance(result, str):
                    # If result is a string, it's just the memory text
                    # We don't have metadata, so skip filtering
                    memories.append({
                        'memory': result,
                        'intent': '',
                        'reward': 1.0,  # Assume high quality if we can't filter
                        'task_index': -1,
                    })
                elif isinstance(result, dict):
                    # mem0 returns: {'id', 'memory', 'metadata', 'score', ...}
                    memory_text = result.get('memory', '')
                    metadata = result.get('metadata', {})

                    # Apply reward filter if specified
                    if filter and 'reward' in filter:
                        min_reward = filter['reward']
                        if metadata.get('reward', 0) < min_reward:
                            continue

                    memories.append({
                        'memory': memory_text,
                        'intent': metadata.get('intent', ''),
                        'reward': metadata.get('reward', 0.0),
                        'task_index': metadata.get('task_index', -1),
                    })

            logger.debug(f"Retrieved {len(memories)} memories for intent: {intent[:50]}...")
            return memories

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []

    def get_stats(self) -> Dict:
        """Get statistics about the memory collection."""
        try:
            all_memories = self.memory.get_all(user_id=self.collection_name)
            return {
                "total_memories": len(all_memories),
                "collection_name": self.collection_name,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"total_memories": 0, "collection_name": self.collection_name}
