import flwr as fl
from typing import Dict, List
import json
from src.embeddings import BioBERTEmbedder
from src.vector_store import QdrantVectorStore
from src.preprocessing import MedicalTextPreprocessor
from src.config import config

class MedicalRAGClient(fl.client.NumPyClient):
    """Flower client for federated medical RAG"""
    
    def __init__(self, client_id: int, data_path: str):
        self.client_id = client_id
        self.data_path = data_path
        
        # Initialize components
        self.embedder = BioBERTEmbedder(config.model.embedding_model)
        self.preprocessor = MedicalTextPreprocessor(
            chunk_size=config.retrieval.chunk_size,
            chunk_overlap=config.retrieval.chunk_overlap
        )
        self.vector_store = QdrantVectorStore(
            host=config.vector_store.host,
            port=config.vector_store.port,
            collection_name=f"{config.vector_store.collection_name}_client_{client_id}"
        )
        
        # Load and index local documents
        self._load_and_index_documents()
    
    def _load_and_index_documents(self):
        """Load and index client's local documents"""
        print(f"Client {self.client_id}: Loading documents...")
        
        # Load documents
        with open(f"{self.data_path}/documents.json", 'r') as f:
            data = json.load(f)
        
        documents = data['documents']
        metadata = data['metadata']
        
        # Preprocess
        chunks = self.preprocessor.chunk_documents(documents)
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        print(f"Client {self.client_id}: Generating embeddings...")
        embeddings = self.embedder.encode(chunk_texts)
        
        # Create vector store collection
        self.vector_store.create_collection(dimension=self.embedder.dimension)
        
        # Add to vector store
        chunk_metadata = [
            {**metadata[chunk['doc_id']], 'chunk_id': chunk['chunk_id']}
            for chunk in chunks
        ]
        self.vector_store.add_documents(chunk_texts, embeddings, chunk_metadata)
        
        print(f"Client {self.client_id}: Indexed {len(chunk_texts)} chunks")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        # Encode query
        query_embedding = self.embedder.encode_query(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        return results