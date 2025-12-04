# measure_hallucination/mitigation/retrieval_augmentation.py
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

class RetrievalAugmenter:
    """Enhanced retrieval strategies to reduce hallucination"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.bm25 = None
        
    def hybrid_retrieval(self, query: str, documents: List[str], 
                        top_k: int = 5, alpha: float = 0.5) -> List[Tuple[str, float]]:
        """
        Hybrid retrieval combining BM25 and semantic search
        
        Args:
            query: User query
            documents: List of documents to search
            top_k: Number of results to return
            alpha: Weight for BM25 (1-alpha for semantic)
            
        Returns:
            List of (document, score) tuples
        """
        if not documents:
            return []
        
        # 1. BM25 retrieval
        bm25_scores = self._bm25_retrieval(query, documents)
        
        # 2. Semantic retrieval
        semantic_scores = self._semantic_retrieval(query, documents)
        
        # 3. Normalize scores
        bm25_normalized = self._normalize_scores(bm25_scores)
        semantic_normalized = self._normalize_scores(semantic_scores)
        
        # 4. Combine scores
        combined_scores = []
        for i, doc in enumerate(documents):
            combined = (alpha * bm25_normalized[i] + 
                       (1 - alpha) * semantic_normalized[i])
            combined_scores.append((doc, combined))
        
        # 5. Sort and return top_k
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        return combined_scores[:top_k]
    
    def rerank_by_relevance(self, query: str, retrieved_docs: List[str], 
                           threshold: float = 0.7) -> List[Tuple[str, float, bool]]:
        """
        Rerank retrieved documents by relevance to query
        
        Args:
            query: Original query
            retrieved_docs: Retrieved documents
            threshold: Minimum relevance threshold
            
        Returns:
            List of (document, relevance_score, is_relevant)
        """
        if not retrieved_docs:
            return []
        
        # Calculate relevance scores
        relevance_scores = []
        query_embedding = self.model.encode(query).reshape(1, -1)
        
        for doc in retrieved_docs:
            doc_embedding = self.model.encode(doc).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
            normalized = (similarity + 1) / 2  # Convert to 0-1
            is_relevant = normalized >= threshold
            relevance_scores.append((doc, normalized, is_relevant))
        
        # Sort by relevance score
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        
        return relevance_scores
    
    def context_expansion(self, query: str, initial_context: str, 
                         knowledge_base: List[str], expansion_factor: int = 2) -> str:
        """
        Expand context with related information
        
        Args:
            query: User query
            initial_context: Initially retrieved context
            knowledge_base: Full knowledge base
            expansion_factor: How many additional documents to include
            
        Returns:
            Expanded context
        """
        if not knowledge_base:
            return initial_context
        
        # Remove initial context from knowledge base if present
        remaining_docs = [doc for doc in knowledge_base if doc != initial_context]
        
        if not remaining_docs:
            return initial_context
        
        # Find related documents
        related_docs = self.hybrid_retrieval(
            query=query,
            documents=remaining_docs,
            top_k=expansion_factor,
            alpha=0.4  # More weight to semantic for expansion
        )
        
        # Combine contexts
        expanded_parts = [initial_context]
        for doc, score in related_docs:
            if score > 0.5:  # Only add if reasonably relevant
                expanded_parts.append(doc)
        
        expanded_context = "\n\n".join(expanded_parts)
        
        return expanded_context
    
    def _bm25_retrieval(self, query: str, documents: List[str]) -> List[float]:
        """BM25 keyword-based retrieval"""
        # Tokenize documents
        tokenized_docs = [doc.split() for doc in documents]
        
        # Initialize BM25 if not done
        if self.bm25 is None or len(self.bm25.doc_freqs) != len(documents):
            self.bm25 = BM25Okapi(tokenized_docs)
        
        # Tokenize query and get scores
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        
        return scores.tolist()
    
    def _semantic_retrieval(self, query: str, documents: List[str]) -> List[float]:
        """Semantic/vector-based retrieval"""
        # Encode query and documents
        query_embedding = self.model.encode(query).reshape(1, -1)
        doc_embeddings = self.model.encode(documents)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        return similarities.tolist()
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)  # All equal
        
        normalized = [(s - min_score) / (max_score - min_score) for s in scores]
        return normalized
