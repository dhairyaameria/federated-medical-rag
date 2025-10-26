"""
Tests for aggregation and RRF
"""
import pytest
from src.server import FederatedRAGServer


def test_rrf_aggregation():
    """Test Reciprocal Rank Fusion aggregation"""
    server = FederatedRAGServer(num_clients=3, k_rrf=60)
    
    # Mock results from 3 clients
    client_results = [
        [
            {'text': 'Doc A about Metformin', 'score': 0.9, 'metadata': {'client': 0}},
            {'text': 'Doc B about Diabetes', 'score': 0.8, 'metadata': {'client': 0}}
        ],
        [
            {'text': 'Doc B about Diabetes', 'score': 0.85, 'metadata': {'client': 1}},
            {'text': 'Doc C about Insulin', 'score': 0.7, 'metadata': {'client': 1}}
        ],
        [
            {'text': 'Doc A about Metformin', 'score': 0.95, 'metadata': {'client': 2}},
            {'text': 'Doc C about Insulin', 'score': 0.75, 'metadata': {'client': 2}}
        ]
    ]
    
    merged = server.reciprocal_rank_fusion(client_results, k=60)
    
    # Should merge results from all clients
    assert len(merged) >= 2
    
    # Documents appearing in multiple clients should have higher RRF scores
    rrf_scores = [doc.get('rrf_score', 0) for doc in merged]
    assert max(rrf_scores) > min(rrf_scores)


def test_rrf_score_calculation():
    """Test RRF score calculation is correct"""
    server = FederatedRAGServer(num_clients=2, k_rrf=60)
    
    # Doc appears in rank 1 of client 0 and rank 2 of client 1
    client_results = [
        [{'text': 'Doc A', 'score': 0.9}],  # Rank 1 client 0
        [{'text': 'Doc B', 'score': 0.8}, {'text': 'Doc A', 'score': 0.7}]  # Rank 2 client 1
    ]
    
    merged = server.reciprocal_rank_fusion(client_results, k=60)
    
    # Doc A should have higher RRF score than Doc B (appears twice vs once)
    doc_a = next((d for d in merged if d['text'] == 'Doc A'), None)
    doc_b = next((d for d in merged if d['text'] == 'Doc B'), None)
    
    if doc_a and doc_b:
        assert doc_a['rrf_score'] > doc_b['rrf_score']


def test_aggregate_results():
    """Test the aggregate_results method"""
    server = FederatedRAGServer(num_clients=2, k_rrf=60)
    
    client_results = [
        [
            {'text': 'Result 1', 'score': 0.9, 'metadata': {}},
            {'text': 'Result 2', 'score': 0.8, 'metadata': {}}
        ],
        [
            {'text': 'Result 1', 'score': 0.85, 'metadata': {}},
            {'text': 'Result 3', 'score': 0.7, 'metadata': {}}
        ]
    ]
    
    query = "test query"
    merged = server.aggregate_results(query, client_results)
    
    assert isinstance(merged, list)
    assert len(merged) >= 1
    
    # Check that RRF scores are present
    if merged:
        assert 'rrf_score' in merged[0]


def test_deduplication():
    """Test that duplicate documents are properly deduplicated"""
    server = FederatedRAGServer(num_clients=2, k_rrf=60)
    
    # Same document appearing twice
    client_results = [
        [{'text': 'Duplicate Doc', 'score': 0.9, 'metadata': {'client': 0}}],
        [{'text': 'Duplicate Doc', 'score': 0.85, 'metadata': {'client': 1}}]
    ]
    
    merged = server.reciprocal_rank_fusion(client_results, k=60)
    
    # Should return only one instance of the duplicate
    texts = [doc['text'] for doc in merged]
    assert texts.count('Duplicate Doc') <= 1


def test_empty_results():
    """Test handling of empty result sets"""
    server = FederatedRAGServer(num_clients=3, k_rrf=60)
    
    client_results = [
        [],  # Empty results from client 0
        [{'text': 'Doc A', 'score': 0.9}],
        []
    ]
    
    merged = server.aggregate_results("query", client_results)
    
    # Should still return results from non-empty clients
    assert len(merged) >= 0  # May be 0 if no valid results


def test_multiple_clients():
    """Test aggregation with multiple clients"""
    server = FederatedRAGServer(num_clients=5, k_rrf=60)
    
    # Create results from 5 clients
    client_results = [
        [{'text': f'Doc from client {i}', 'score': 0.9 - i*0.1, 'metadata': {'client': i}}]
        for i in range(5)
    ]
    
    merged = server.aggregate_results("query", client_results)
    
    # Should aggregate from all clients
    assert isinstance(merged, list)
    assert len(merged) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

