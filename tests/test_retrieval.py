"""
Tests for retrieval functionality
"""
import pytest
import numpy as np
from src.retrieval import MedicalRetriever, HybridRetriever
from src.embeddings import BioBERTEmbedder


def test_dense_retrieval():
    """Test dense retrieval using vector similarity"""
    embedder = BioBERTEmbedder()
    retriever = MedicalRetriever(embedder.model)
    
    # Sample documents
    documents = [
        "Metformin is used to treat type 2 diabetes",
        "Diabetes treatment options include insulin therapy",
        "High blood pressure affects millions of people",
        "The side effects of Metformin include nausea and vomiting"
    ]
    
    embeddings = embedder.encode(documents)
    
    # Search for metformin-related content
    results = retriever.dense_retrieval(
        query="What are Metformin side effects?",
        documents=documents,
        embeddings=embeddings,
        top_k=2
    )
    
    assert len(results) == 2
    assert results[0]['score'] > 0
    # Should retrieve the document about side effects
    assert "Metformin" in results[0]['text']


def test_reciprocal_rank_fusion():
    """Test RRF aggregation of multiple result lists"""
    retriever = MedicalRetriever(None)
    
    # Mock result lists from different retrieval methods
    results1 = [
        {'text': 'Doc A', 'score': 0.9, 'index': 0},
        {'text': 'Doc B', 'score': 0.8, 'index': 1}
    ]
    results2 = [
        {'text': 'Doc B', 'score': 0.85, 'index': 1},
        {'text': 'Doc C', 'score': 0.7, 'index': 2}
    ]
    
    merged = retriever.reciprocal_rank_fusion([results1, results2], k=60)
    
    # Doc B should have higher RRF score (appears in both lists)
    assert len(merged) >= 1
    assert merged[0].get('rrf_score', 0) > 0


def test_expand_medical_query():
    """Test medical query expansion"""
    retriever = MedicalRetriever(None)
    
    expanded = retriever.expand_medical_query("diabetes treatment")
    
    assert len(expanded) >= 1
    assert "diabetes treatment" in expanded or any("diabetes" in q for q in expanded)


def test_rerank_results():
    """Test result reranking"""
    retriever = MedicalRetriever(None)
    
    documents = [
        {'text': 'Doc A', 'score': 0.5},
        {'text': 'Doc B', 'score': 0.9},
        {'text': 'Doc C', 'score': 0.7}
    ]
    
    reranked = retriever.rerank_results(
        query="test query",
        documents=documents,
        top_k=2
    )
    
    assert len(reranked) == 2
    # Should be sorted by score
    assert reranked[0]['score'] >= reranked[1]['score']


def test_hybrid_retrieval():
    """Test hybrid retrieval combining dense and sparse methods"""
    embedder = BioBERTEmbedder()
    hybrid = HybridRetriever(embedder.model)
    
    documents = [
        "Metformin side effects include nausea",
        "High blood pressure medication",
        "Diabetes type 2 treatment with Metformin"
    ]
    
    embeddings = embedder.encode(documents)
    
    results = hybrid.retrieve(
        query="Metformin side effects",
        documents=documents,
        embeddings=embeddings,
        top_k=2
    )
    
    assert len(results) <= 2
    assert all(r['score'] >= 0 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

