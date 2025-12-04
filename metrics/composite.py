# measure_hallucination/metrics/composite.py
from typing import Dict
from .faithfulness import FaithfulnessMetrics
from .relevance import RelevanceMetrics
from .consistency import ConsistencyMetrics

class CompositeMetrics:
    """Combine all metrics into comprehensive hallucination score"""
    
    def __init__(self):
        self.faithfulness = FaithfulnessMetrics()
        self.relevance = RelevanceMetrics()
        self.consistency = ConsistencyMetrics()
    
    def calculate_all_metrics(self, query: str, context: str, answer: str, 
                             multiple_answers: list = None) -> Dict:
        """
        Calculate comprehensive hallucination metrics
        
        Args:
            query: User query
            context: Retrieved context
            answer: Generated answer
            multiple_answers: Optional list of answers for consistency check
            
        Returns:
            Complete metrics dictionary
        """
        # Calculate individual metrics
        faithfulness = self.faithfulness.faithfulness_score(answer, context)
        context_rel = self.relevance.context_relevance(query, context)
        answer_rel = self.relevance.answer_relevance(query, answer)
        
        # Calculate consistency if multiple answers provided
        if multiple_answers and len(multiple_answers) > 1:
            consistency = self.consistency.self_consistency(multiple_answers)
        else:
            consistency = {'consistency_score': 1.0, 'consistency_level': 'HIGH'}
        
        # Calculate composite hallucination score
        hall_score = self._calculate_hallucination_score(
            faithfulness['faithfulness'],
            context_rel['semantic_relevance'],
            answer_rel['overall_relevance'],
            consistency['consistency_score']
        )
        
        return {
            'faithfulness': faithfulness,
            'context_relevance': context_rel,
            'answer_relevance': answer_rel,
            'consistency': consistency,
            'hallucination_score': hall_score,
            'risk_assessment': self._assess_risk(hall_score),
            'recommendations': self._generate_recommendations(
                faithfulness, context_rel, answer_rel
            )
        }
    
    def _calculate_hallucination_score(self, faithfulness: float, 
                                      context_rel: float, 
                                      answer_rel: float, 
                                      consistency: float) -> float:
        """Calculate weighted hallucination score (0-1, higher = more hallucination)"""
        # Weighted combination - adjust weights based on importance
        weights = {
            'faithfulness': 0.4,      # Most important - is answer grounded?
            'context_rel': 0.3,       # Important - is context relevant?
            'answer_rel': 0.2,        # Important - does answer address query?
            'consistency': 0.1        # Less important but still valuable
        }
        
        # Invert faithfulness and consistency (higher = less hallucination)
        hall_faithfulness = 1 - faithfulness
        hall_consistency = 1 - consistency
        
        # For relevance, low score indicates potential hallucination
        hall_context_rel = 1 - context_rel
        hall_answer_rel = 1 - answer_rel
        
        composite = (
            weights['faithfulness'] * hall_faithfulness +
            weights['context_rel'] * hall_context_rel +
            weights['answer_rel'] * hall_answer_rel +
            weights['consistency'] * hall_consistency
        )
        
        return float(composite)
    
    def _assess_risk(self, hall_score: float) -> Dict:
        """Assess hallucination risk level"""
        if hall_score < 0.2:
            level = 'VERY_LOW'
            color = '🟢'
        elif hall_score < 0.4:
            level = 'LOW'
            color = '🟡'
        elif hall_score < 0.6:
            level = 'MEDIUM'
            color = '🟠'
        elif hall_score < 0.8:
            level = 'HIGH'
            color = '🔴'
        else:
            level = 'VERY_HIGH'
            color = '💀'
        
        return {
            'level': level,
            'score': hall_score,
            'color': color,
            'thresholds': {
                'VERY_LOW': (0.0, 0.2),
                'LOW': (0.2, 0.4),
                'MEDIUM': (0.4, 0.6),
                'HIGH': (0.6, 0.8),
                'VERY_HIGH': (0.8, 1.0)
            }
        }
    
    def _generate_recommendations(self, faithfulness: Dict, 
                                 context_rel: Dict, 
                                 answer_rel: Dict) -> List[str]:
        """Generate actionable recommendations based on metrics"""
        recommendations = []
        
        # Faithfulness recommendations
        if faithfulness['faithfulness'] < 0.7:
            recommendations.append(
                f"⚠️ Improve answer grounding: Only {faithfulness['supported_claims']}/"
                f"{faithfulness['total_claims']} claims are well-supported"
            )
        
        # Context relevance recommendations
        if context_rel['semantic_relevance'] < 0.6:
            recommendations.append(
                f"🔍 Improve context retrieval: Context relevance is "
                f"{context_rel['semantic_relevance']:.1%} ({context_rel['relevance_level']})"
            )
        
        # Answer relevance recommendations
        if answer_rel['overall_relevance'] < 0.7:
            recommendations.append(
                f"🎯 Improve answer relevance: Answer doesn't fully address the query "
                f"(relevance: {answer_rel['overall_relevance']:.1%})"
            )
        
        if not recommendations:
            recommendations.append("✅ All metrics within acceptable ranges")
        
        return recommendations
