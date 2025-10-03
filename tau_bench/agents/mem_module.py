from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from openai import OpenAI
from qdrant_client.http import models as rest


from qdrant_client.models import Filter, FieldCondition, Range

# hits = client.query_points(
#     collection_name="my_collection",
#     query=query_vector,
#     query_filter=Filter(
#         must=[  # These conditions are required for search results
#             FieldCondition(
#                 key='rand_number',  # Condition based on values of `rand_number` field.
#                 range=Range(
#                     gte=3  # Select only those results where `rand_number` >= 3
#                 )
#             )
#         ]
#     ),
#     limit=5  # Return 5 closest points
# )

class MemModule:
    def __init__(self, collection_name: str, delete_existing: bool = False):
        self.client = QdrantClient("http://localhost:6333")
        self.openai_client = OpenAI()
        ## check if collection exists
        if delete_existing:
            if self.client.collection_exists(collection_name=collection_name):
                print("num points before deletion: ", self.client.count(collection_name=collection_name))
                self.client.delete_collection(collection_name=collection_name)
                print("num points after deletion: ", self.client.count(collection_name=collection_name))
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(collection_name=collection_name, 
                                          vectors_config=VectorParams(size=1536, 
                                                                      distance=Distance.COSINE))
        else:
            self.collection = self.client.get_collection(collection_name=collection_name)

        self.collection_name = collection_name
        
    
    def embed(self, text: str):
        return self.openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        
        
    def add_memory(self, structured_memory: dict):
        traj_memory = structured_memory["memory"]
        intent = structured_memory["intent"]
        reward = structured_memory["reward"]
        task_index = structured_memory["task_index"]
        vector = self.embed(intent).data[0].embedding
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                rest.PointStruct(
                    id=task_index,
                    vector=vector,
                    payload={"task_index": task_index, "reward": reward, "memory": traj_memory, "intent": intent}
                )
            ]
        )
        
        
    def retrieve_memory(self, intent: str, filter: dict = None, limit: int = 2):
        if filter is not None:
            filter = Filter(
                must=[FieldCondition(key='reward', range=Range(gte=filter['reward']))]
            )
        else:
            filter = None
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=self.embed(intent).data[0].embedding,
            limit=limit,
            query_filter=filter
            )
        return [hit.payload for hit in hits.points]

        
        
        
        