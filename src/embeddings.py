from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class BioBERTEmbedder:
    """BioBERT embedding model for medical text"""
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query"""
        return self.model.encode([query], convert_to_numpy=True)[0]