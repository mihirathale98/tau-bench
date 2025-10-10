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
                    "host": "localhost",
                    "port": 6333,
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small"
                }
            }
        }

        # Initialize mem0
        self.memory = Memory.from_config(default_config)

        # Clear existing memories if requested
        if delete_existing:
            print("Deleting existing memories from mem0")
            try:
                self.memory.delete_all(agent_id="our_agent")
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
        trajectory = structured_memory["trajectory"]
        intent = structured_memory["intent"]
        reward = structured_memory["reward"]
        task_index = structured_memory["task_index"]
        agent_id = structured_memory["agent_id"]
        

        # mem0 stores memories with metadata
        # We use collection_name as user_id to organize by environment
        print("Calling add now ")
        self.memory.add(
            messages=trajectory,  # The actual memory content
            agent_id=agent_id,
            metadata={
                "intent": intent,
                "reward": reward,
                "task_index": task_index,
            },
            memory_type="procedural_memory"
        )
        print("Added memory")
        ## print what got added filter with task_index
        try:
            print(f"Memory: {self.memory.get_all(agent_id=agent_id, filters={'task_index': task_index})}")
        except Exception as e:
            print("Error ", e)
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
                agent_id="our_agent",
                limit=limit
            )['results']
            memory_string = ""
            for result in results:
                intent = result.get("metadata", {}).get("intent", "")
                memory_string += f"**Intent:** {intent}\n"
                memory_string += f"**Memory:** {result.get("memory", "")}\n\n"
            return memory_string

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []

    def get_stats(self) -> Dict:
        """Get statistics about the memory collection."""
        try:
            all_memories = self.memory.get_all(agent_id=self.collection_name)
            return {
                "total_memories": len(all_memories),
                "collection_name": self.collection_name,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"total_memories": 0, "collection_name": self.collection_name}
