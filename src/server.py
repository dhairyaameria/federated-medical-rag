import flwr as fl
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np

class FederatedRAGServer:
    """Flower server for federated RAG aggregation"""
    
    def __init__(self, num_clients: int = 3, k_rrf: int = 60):
        self.num_clients = num_clients
        self.k_rrf = k_rrf
    
    def reciprocal_rank_fusion(self, 
                               client_results: List[List[Dict]], 
                               k: int = 60) -> List[Dict]:
        """
        Aggregate results from multiple clients using Reciprocal Rank Fusion
        
        Args:
            client_results: List of result lists from each client
            k: RRF constant (default 60)
        
        Returns:
            Merged and ranked results
        """
        doc_scores = defaultdict(float)
        doc_data = {}
        
        # Calculate RRF scores
        for client_results_list in client_results:
            for rank, doc in enumerate(client_results_list, start=1):
                doc_id = doc['text'][:100]  # Use text prefix as ID
                rrf_score = 1.0 / (k + rank)
                doc_scores[doc_id] += rrf_score
                
                if doc_id not in doc_data:
                    doc_data[doc_id] = doc
        
        # Sort by aggregated scores
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top documents with scores
        merged_results = [
            {
                **doc_data[doc_id],
                'rrf_score': score
            }
            for doc_id, score in sorted_docs
        ]
        
        return merged_results
    
    def aggregate_results(self, query: str, client_results: List[List[Dict]]) -> List[Dict]:
        """Aggregate results from all clients"""
        print(f"\nAggregating results from {len(client_results)} clients...")
        
        # Use RRF to merge results
        merged_results = self.reciprocal_rank_fusion(client_results, k=self.k_rrf)
        
        print(f"Merged into {len(merged_results)} unique documents")
        return merged_results