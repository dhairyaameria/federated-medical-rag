"""
Tests for embedding generation
"""
import pytest
import numpy as np
from src.embeddings import BioBERTEmbedder


def test_embedding_model_loading():
    """Test BioBERT model loads correctly"""
    embedder = BioBERTEmbedder()
    
    assert embedder.model is not None
    assert embedder.dimension > 0
    assert embedder.dimension == 768  # BioBERT default dimension


def test_encode_single_query():
    """Test encoding a single query"""
    embedder = BioBERTEmbedder()
    
    query = "What are the side effects of Metformin?"
    embedding = embedder.encode_query(query)
    
    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedder.dimension,)


def test_encode_multiple_documents():
    """Test encoding multiple documents"""
    embedder = BioBERTEmbedder()
    
    documents = [
        "Metformin is used for diabetes treatment",
        "Hypertension affects many adults",
        "Aspirin can prevent heart attacks"
    ]
    
    embeddings = embedder.encode(documents)
    
    assert embeddings is not None
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(documents), embedder.dimension)
    assert embeddings.shape[0] == 3


def test_embedding_consistency():
    """Test that same query produces same embedding"""
    embedder = BioBERTEmbedder()
    
    query = "test query"
    
    embedding1 = embedder.encode_query(query)
    embedding2 = embedder.encode_query(query)
    
    # Embeddings should be very similar (same model, same input)
    np.testing.assert_allclose(embedding1, embedding2, rtol=1e-5)


def test_embedding_dimensions():
    """Test embedding dimensions are correct"""
    embedder = BioBERTEmbedder()
    
    queries = ["query 1", "query 2", "query 3"]
    documents = ["doc 1", "doc 2"]
    
    query_embeddings = [embedder.encode_query(q) for q in queries]
    doc_embeddings = embedder.encode(documents)
    
    # All embeddings should have same dimension
    for q_emb in query_embeddings:
        assert q_emb.shape == (embedder.dimension,)
    
    assert doc_embeddings.shape == (len(documents), embedder.dimension)


def test_medical_term_embeddings():
    """Test embeddings for medical terminology"""
    embedder = BioBERTEmbedder()
    
    medical_terms = [
        "diabetes mellitus",
        "metabolic syndrome",
        "hypertension",
        "myocardial infarction"
    ]
    
    embeddings = embedder.encode(medical_terms)
    
    assert embeddings.shape[0] == len(medical_terms)
    assert embeddings.shape[1] == embedder.dimension
    
    # Check embeddings are not all zeros
    assert np.any(embeddings != 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

