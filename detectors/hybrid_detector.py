# measure_hallucination/detectors/hybrid_detector.py
from typing import Dict
from ..metrics.faithfulness import FaithfulnessMetrics
from ..metrics.relevance import RelevanceMetrics
from ..detectors.rule_based import RuleBasedDetector
from ..detectors.ml_based import MLBasedDetector

class HybridDetector:
    """Hybrid hallucination detector combining rule-based and ML approaches"""
    
    def __init__(self, ml_model_path: str = None):
        self.faithfulness = FaithfulnessMetrics()
        self.relevance = RelevanceMetrics()
        self.rule_based = RuleBasedDetector()
        self.ml_based = MLBasedDetector(ml_model_path)
        
        # Weights for combining different detectors
        self.weights = {
            'faithfulness': 0.35,
            'relevance': 0.25,
            'rule_based': 0.20,
            'ml_based': 0.20
        }
    
    def detect(self, query: str, context: str, answer: str) -> Dict:
        """
        Hybrid hallucination detection
        
        Args:
            query: User query
            context: Retrieved context
            answer: Generated answer
            
        Returns:
            Comprehensive detection results
        """
        results = {
            'components': {},
            'final_decision': {},
            'explanation': '',
            'confidence': 0.0
        }
        
        # 1. Faithfulness detection
        faithfulness_results = self.faithfulness.faithfulness_score(answer, context)
        results['components']['faithfulness'] = faithfulness_results
        
        # 2. Relevance detection
        context_rel = self.relevance.context_relevance(query, context)
        answer_rel = self.relevance.answer_relevance(query, answer)
        results['components']['relevance'] = {
            'context_relevance': context_rel,
            'answer_relevance': answer_rel
        }
        
        # 3. Rule-based detection
        rule_results = self.rule_based.detect(answer, context)
        results['components']['rule_based'] = rule_results
        
        # 4. ML-based detection
        ml_results = self.ml_based.detect(query, context, answer)
        results['components']['ml_based'] = ml_results
        
        # 5. Combine scores
        combined_score = self._combine_scores(
            faithfulness_results['faithfulness'],
            context_rel['semantic_relevance'],
            answer_rel['overall_relevance'],
            rule_results['overall_score'],
            ml_results.get('confidence', 0.5) if 'error' not in ml_results else 0.5
        )
        
        # 6. Make final decision
        final_decision = self._make_decision(combined_score, results['components'])
        
        results['final_decision'] = final_decision
        results['combined_score'] = combined_score
        results['explanation'] = self._generate_explanation(results['components'])
        results['confidence'] = final_decision['confidence']
        
        return results
    
    def _combine_scores(self, faithfulness: float, context_rel: float, 
                       answer_rel: float, rule_score: float, ml_confidence: float) -> float:
        """
        Combine scores from different detectors
        
        Note: Higher scores indicate more hallucination for rule-based,
        but less hallucination for others. We need to normalize.
        """
        # Normalize scores to 0-1 where 1 = high hallucination
        norm_faithfulness = 1 - faithfulness  # Low faithfulness = high hallucination
        norm_context_rel = 1 - context_rel    # Low relevance = high hallucination
        norm_answer_rel = 1 - answer_rel      # Low relevance = high hallucination
        
        # Rule score is already high = more hallucination
        norm_rule = rule_score
        
        # ML confidence: high confidence in hallucination = high score
        # Assuming ml_confidence is probability of hallucination
        norm_ml = ml_confidence
        
        # Weighted combination
        combined = (
            self.weights['faithfulness'] * norm_faithfulness +
            self.weights['relevance'] * ((norm_context_rel + norm_answer_rel) / 2) +
            self.weights['rule_based'] * norm_rule +
            self.weights['ml_based'] * norm_ml
        )
        
        return min(max(combined, 0.0), 1.0)
    
    def _make_decision(self, combined_score: float, components: Dict) -> Dict:
        """Make final hallucination decision"""
        # Check individual component flags
        faithfulness_risk = components['faithfulness']['risk_level']
        rule_risk = components['rule_based']['risk_level']
        
        # ML prediction if available
        ml_prediction = components['ml_based'].get('prediction', 'VALID')
        
        # Determine overall risk
        risk_factors = []
        
        if faithfulness_risk == 'HIGH':
            risk_factors.append('Poor answer grounding')
        if rule_risk == 'HIGH':
            risk_factors.append('Patterns of hallucination')
        if ml_prediction == 'HALLUCINATION':
            risk_factors.append('ML model prediction')
        if combined_score > 0.6:
            risk_factors.append('High combined score')
        
        # Final classification
        if len(risk_factors) >= 2 or combined_score > 0.7:
            decision = 'HALLUCINATION'
            confidence = combined_score
            severity = 'HIGH'
        elif len(risk_factors) >= 1 or combined_score > 0.4:
            decision = 'POTENTIAL_HALLUCINATION'
            confidence = combined_score
            severity = 'MEDIUM'
        else:
            decision = 'VALID'
            confidence = 1 - combined_score
            severity = 'LOW'
        
        return {
            'decision': decision,
            'confidence': float(confidence),
            'severity': severity,
            'risk_factors': risk_factors,
            'combined_score': float(combined_score),
            'thresholds_used': {
                'high_risk': 0.7,
                'medium_risk': 0.4
            }
        }
    
    def _generate_explanation(self, components: Dict) -> str:
        """Generate human-readable explanation"""
        explanation_parts = []
        
        # Faithfulness explanation
        faith = components['faithfulness']
        explanation_parts.append(
            f"Faithfulness: {faith['supported_claims']}/{faith['total_claims']} "
            f"claims supported ({faith['faithfulness']:.1%})"
        )
        
        # Rule-based explanation
        rule = components['rule_based']
        if rule['patterns_found']:
            explanation_parts.append(
                f"Rule-based: Found {len(rule['patterns_found'])} suspicious patterns"
            )
        
        # ML explanation
        ml = components['ml_based']
        if 'prediction' in ml:
            explanation_parts.append(
                f"ML prediction: {ml['prediction']} with {ml.get('confidence', 0):.1%} confidence"
            )
        
        return ". ".join(explanation_parts)
