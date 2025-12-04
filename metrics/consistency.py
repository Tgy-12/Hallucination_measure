# measure_hallucination/metrics/consistency.py
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ConsistencyMetrics:
    """Calculate self-consistency and factual consistency metrics"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def self_consistency(self, answers: List[str]) -> Dict:
        """
        Calculate consistency between multiple answers to same query
        
        Args:
            answers: List of answers generated for the same query
            
        Returns:
            Consistency scores and analysis
        """
        if len(answers) < 2:
            return {
                'consistency_score': 1.0,
                'pairwise_similarities': [],
                'variance': 0.0,
                'consistency_level': 'HIGH'
            }
        
        # Encode all answers
        embeddings = self.model.encode(answers)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Get upper triangle (excluding diagonal)
        upper_tri = similarities[np.triu_indices(len(answers), k=1)]
        
        if len(upper_tri) == 0:
            avg_similarity = 1.0
        else:
            avg_similarity = float(np.mean(upper_tri))
        
        # Normalize to 0-1
        normalized_consistency = (avg_similarity + 1) / 2
        
        # Calculate variance
        variance = float(np.var(upper_tri)) if len(upper_tri) > 0 else 0.0
        
        # Determine consistency level
        if normalized_consistency >= 0.8:
            level = 'HIGH'
        elif normalized_consistency >= 0.6:
            level = 'MEDIUM'
        else:
            level = 'LOW'
        
        return {
            'consistency_score': normalized_consistency,
            'raw_similarity': float(avg_similarity),
            'pairwise_similarities': upper_tri.tolist(),
            'variance': variance,
            'consistency_level': level,
            'num_answers': len(answers)
        }
    
    def factual_consistency(self, answer1: str, answer2: str) -> Dict:
        """Check if two answers are factually consistent"""
        embeddings = self.model.encode([answer1, answer2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        normalized = (similarity + 1) / 2
        
        return {
            'similarity_score': float(normalized),
            'is_consistent': normalized >= 0.7,
            'threshold': 0.7
        }
