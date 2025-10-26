"""
Advanced retrieval logic for the federated medical RAG system
Includes reranking, hybrid search, and query expansion capabilities
"""
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class MedicalRetriever:
    """Advanced retrieval with reranking and hybrid search"""
    
    def __init__(self, embedding_model, chunk_size=512):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
    
    def dense_retrieval(self, query: str, documents: List[str], 
                       embeddings: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        Dense retrieval using vector similarity
        
        Args:
            query: Search query
            documents: List of document texts
            embeddings: Document embeddings
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents with scores
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarities
        similarities = np.dot(embeddings, query_embedding)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': documents[idx],
                'score': float(similarities[idx]),
                'index': int(idx)
            })
        
        return results
    
    def reciprocal_rank_fusion(self, result_lists: List[List[Dict]], 
                               k: int = 60) -> List[Dict]:
        """
        Combine results from multiple retrieval methods using RRF
        
        Args:
            result_lists: List of result lists from different methods
            k: RRF constant (default 60)
            
        Returns:
            Merged and reranked results
        """
        doc_scores = {}
        doc_data = {}
        
        # Calculate RRF scores
        for results in result_lists:
            for rank, doc in enumerate(results, start=1):
                doc_id = hash(doc['text'])  # Unique document identifier
                rrf_score = 1.0 / (k + rank)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                    doc_data[doc_id] = doc
                
                doc_scores[doc_id] += rrf_score
        
        # Sort by aggregated RRF scores
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return documents with RRF scores
        merged_results = [
            {**doc_data[doc_id], 'rrf_score': score}
            for doc_id, score in sorted_docs
        ]
        
        return merged_results
    
    def expand_medical_query(self, query: str) -> List[str]:
        """
        Expand medical query with related terms
        Basic implementation - can be enhanced with medical ontologies
        
        Args:
            query: Original query
            
        Returns:
            List of expanded query variations
        """
        # Simple synonyms for common medical terms
        expansions = {
            'diabetes': ['diabetes mellitus', 'diabetic', 'blood sugar'],
            'hypertension': ['high blood pressure', 'HTN', 'blood pressure elevation'],
            'cancer': ['neoplasm', 'malignancy', 'tumor'],
            'heart attack': ['myocardial infarction', 'MI', 'cardiac arrest'],
        }
        
        queries = [query]
        
        query_lower = query.lower()
        for term, synonyms in expansions.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded = query_lower.replace(term, synonym)
                    if expanded not in queries:
                        queries.append(expanded)
        
        return queries
    
    def rerank_results(self, query: str, documents: List[Dict], 
                     top_k: int = 5) -> List[Dict]:
        """
        Rerank results using cross-encoder or other methods
        
        Args:
            query: Search query
            documents: Retrieved documents
            top_k: Final number of results
            
        Returns:
            Reranked documents
        """
        # For now, just return top-k based on existing scores
        # In production, use a cross-encoder model for reranking
        
        sorted_docs = sorted(
            documents,
            key=lambda x: x.get('rrf_score', x.get('score', 0)),
            reverse=True
        )
        
        return sorted_docs[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranking for improved precision"""
    
    def __init__(self, model_name: str = None):
        # In production, load a trained cross-encoder
        self.model = None
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: Search query
            documents: Retrieved documents
            top_k: Number of results to return
            
        Returns:
            Reranked documents with updated scores
        """
        if not self.model:
            # Lazy loading - would use sentence-transformers CrossEncoder
            # from sentence_transformers import CrossEncoder
            # self.model = CrossEncoder(self.model_name)
            pass
        
        # Simple fallback: use existing scores
        sorted_docs = sorted(
            documents,
            key=lambda x: x.get('score', 0),
            reverse=True
        )
        
        return sorted_docs[:top_k]


class HybridRetriever:
    """Combine dense and sparse retrieval"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.alpha = 0.7  # Dense retrieval weight
        self.beta = 0.3   # Sparse retrieval weight
    
    def retrieve(self, query: str, documents: List[str], 
                embeddings: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        Hybrid retrieval combining dense and sparse methods
        
        Args:
            query: Search query
            documents: Document texts
            embeddings: Document embeddings
            top_k: Number of results
            
        Returns:
            Combined retrieval results
        """
        # Dense retrieval (already implemented in MedicalRetriever)
        retriever = MedicalRetriever(self.embedding_model)
        dense_results = retriever.dense_retrieval(query, documents, embeddings, top_k)
        
        # Sparse retrieval (BM25) - simplified version
        sparse_results = self._sparse_retrieval(query, documents, top_k)
        
        # Combine using weighted scoring
        combined = self._combine_scores(dense_results, sparse_results)
        
        return combined
    
    def _sparse_retrieval(self, query: str, documents: List[str], 
                         top_k: int) -> List[Dict]:
        """Simplified sparse retrieval using keyword matching"""
        query_terms = set(query.lower().split())
        
        results = []
        for idx, doc in enumerate(documents):
            doc_terms = set(doc.lower().split())
            
            # Jaccard similarity as sparse score
            intersection = len(query_terms & doc_terms)
            union = len(query_terms | doc_terms)
            score = intersection / union if union > 0 else 0
            
            results.append({
                'text': doc,
                'score': score,
                'index': idx
            })
        
        # Sort and return top-k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _combine_scores(self, dense_results: List[Dict], 
                       sparse_results: List[Dict]) -> List[Dict]:
        """Combine dense and sparse scores"""
        # Normalize scores to [0, 1]
        def normalize(scores):
            if not scores:
                return scores
            max_score = max(s['score'] for s in scores)
            return [{**s, 'score': s['score']/max_score if max_score > 0 else 0} 
                   for s in scores]
        
        dense_normalized = normalize(dense_results)
        sparse_normalized = normalize(sparse_results)
        
        # Combine scores
        combined = {}
        for result in dense_normalized:
            idx = result['index']
            combined[idx] = {
                **result,
                'score': result['score'] * self.alpha
            }
        
        for result in sparse_normalized:
            idx = result['index']
            if idx in combined:
                combined[idx]['score'] += result['score'] * self.beta
            else:
                combined[idx] = {
                    **result,
                    'score': result['score'] * self.beta
                }
        
        # Sort and return
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return sorted_results

