import re
from typing import List, Tuple, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)

class FaithfulnessMetrics:
    """Calculate faithfulness scores for generated answers"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.vectorizer = TfidfVectorizer()
        
    def extract_claims(self, text: str) -> List[str]:
        """Extract atomic claims from text"""
        sentences = sent_tokenize(text)
        claims = []
        for sent in sentences:
            # Simple claim extraction - can be enhanced
            if len(sent.split()) > 3:  # Minimum 4 words for a meaningful claim
                claims.append(sent.strip())
        return claims
    
    def calculate_claim_support(self, claim: str, context: str) -> float:
        """Calculate how well a claim is supported by context"""
        # Semantic similarity
        claim_embedding = self.model.encode(claim, convert_to_tensor=True)
        context_embedding = self.model.encode(context, convert_to_tensor=True)
        
        # Cosine similarity
        similarity = np.dot(claim_embedding, context_embedding) / (
            np.linalg.norm(claim_embedding) * np.linalg.norm(context_embedding)
        )
        
        # Keyword overlap
        claim_words = set(claim.lower().split())
        context_words = set(context.lower().split())
        keyword_overlap = len(claim_words.intersection(context_words)) / max(len(claim_words), 1)
        
        # Combined score (weighted)
        combined_score = 0.7 * similarity + 0.3 * keyword_overlap
        return float(combined_score)
    
    def faithfulness_score(self, answer: str, context: str, threshold: float = 0.7) -> Dict:
        """
        Calculate faithfulness score for an answer given context
        
        Args:
            answer: Generated answer
            context: Source context
            threshold: Minimum similarity to consider claim supported
            
        Returns:
            Dictionary with scores and analysis
        """
        claims = self.extract_claims(answer)
        
        if not claims:
            return {
                'faithfulness': 0.0,
                'supported_claims': 0,
                'total_claims': 0,
                'claim_scores': [],
                'risk_level': 'HIGH'
            }
        
        claim_scores = []
        supported_claims = 0
        
        for claim in claims:
            score = self.calculate_claim_support(claim, context)
            claim_scores.append(score)
            if score >= threshold:
                supported_claims += 1
        
        faithfulness = supported_claims / len(claims) if claims else 0.0
        
        # Determine risk level
        if faithfulness >= 0.8:
            risk = 'LOW'
        elif faithfulness >= 0.5:
            risk = 'MEDIUM'
        else:
            risk = 'HIGH'
        
        return {
            'faithfulness': faithfulness,
            'supported_claims': supported_claims,
            'total_claims': len(claims),
            'claim_scores': claim_scores,
            'risk_level': risk,
            'average_support_score': np.mean(claim_scores) if claim_scores else 0.0
        }
