# measure_hallucination/metrics/relevance.py
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import bert_score

class RelevanceMetrics:
    """Calculate context and answer relevance scores"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    def context_relevance(self, query: str, context: str) -> Dict:
        """Calculate how relevant context is to query"""
        # Semantic relevance
        query_embed = self.model.encode(query).reshape(1, -1)
        context_embed = self.model.encode(context).reshape(1, -1)
        semantic_score = cosine_similarity(query_embed, context_embed)[0][0]
        
        # Normalize to 0-1 range (cosine similarity returns -1 to 1)
        normalized_score = (semantic_score + 1) / 2
        
        return {
            'semantic_relevance': float(normalized_score),
            'raw_cosine': float(semantic_score),
            'relevance_level': self._get_relevance_level(normalized_score)
        }
    
    def answer_relevance(self, query: str, answer: str) -> Dict:
        """Calculate how relevant answer is to query"""
        # ROUGE scores
        rouge_scores = self.rouge_scorer.score(query, answer)
        
        # Semantic similarity
        query_embed = self.model.encode(query).reshape(1, -1)
        answer_embed = self.model.encode(answer).reshape(1, -1)
        semantic_score = cosine_similarity(query_embed, answer_embed)[0][0]
        normalized_semantic = (semantic_score + 1) / 2
        
        # BERTScore
        P, R, F1 = bert_score.score([answer], [query], lang='en', verbose=False)
        
        return {
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure,
            'semantic_relevance': float(normalized_semantic),
            'bertscore_precision': float(P.mean()),
            'bertscore_recall': float(R.mean()),
            'bertscore_f1': float(F1.mean()),
            'overall_relevance': float((normalized_semantic + F1.mean()) / 2)
        }
    
    def _get_relevance_level(self, score: float) -> str:
        if score >= 0.8:
            return 'HIGH'
        elif score >= 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'
