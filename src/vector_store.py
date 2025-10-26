from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict
import numpy as np

class QdrantVectorStore:
    """Qdrant vector database manager"""
    
    def __init__(self, host: str = "localhost", port: int = 6333, 
                 collection_name: str = "medical_docs"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
    
    def create_collection(self, dimension: int = 768):
        """Create vector collection if it doesn't exist"""
        try:
            # Check if collection exists first
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                # Collection exists - silently skip
                return
            
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )
            print(f"✓ Created new collection: {self.collection_name}")
        except Exception as e:
            # Only print actual errors
            if "already exists" not in str(e).lower():
                print(f"Warning: {e}")
    
    def add_documents(self, documents: List[str], 
                     embeddings: np.ndarray, 
                     metadata: List[Dict]):
        """Add documents with embeddings to vector store"""
        points = []
        for idx, (doc, emb, meta) in enumerate(zip(documents, embeddings, metadata)):
            point = PointStruct(
                id=idx,
                vector=emb.tolist(),
                payload={
                    'text': doc,
                    **meta
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Added {len(points)} documents to {self.collection_name}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        return [
            {
                'text': hit.payload['text'],
                'score': hit.score,
                'metadata': {k: v for k, v in hit.payload.items() if k != 'text'}
            }
            for hit in results
        ]