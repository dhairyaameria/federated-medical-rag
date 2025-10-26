from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class ModelConfig:
    """Model configuration"""
    embedding_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    llm_model: str = "gpt-4"
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.1
    max_tokens: int = 1024

@dataclass
class RetrievalConfig:
    """Retrieval configuration"""
    top_k: int = 10
    similarity_threshold: float = 0.7
    chunk_size: int = 512
    chunk_overlap: int = 50
    rerank: bool = True

@dataclass
class FlowerConfig:
    """Flower federated learning configuration"""
    num_clients: int = 3
    num_rounds: int = 1  # For RAG, typically 1 round per query
    server_address: str = "localhost:8080"
    k_rrf: int = 60  # Reciprocal Rank Fusion parameter

@dataclass
class VectorStoreConfig:
    """Vector store configuration"""
    store_type: str = "qdrant"
    collection_name: str = "medical_documents"
    host: str = "localhost"
    port: int = 6333
    dimension: int = 768  # BioBERT dimension

class Config:
    """Main configuration class"""
    def __init__(self):
        self.model = ModelConfig()
        self.retrieval = RetrievalConfig()
        self.flower = FlowerConfig()
        self.vector_store = VectorStoreConfig()
        
        # Set API keys from environment
        self.model.llm_api_key = os.getenv("OPENAI_API_KEY")

# Global config instance
config = Config()