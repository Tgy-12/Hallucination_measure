# measure_hallucination/mitigation/post_processing.py
from typing import Dict, List, Tuple, Optional
import re
from ..metrics.faithfulness import FaithfulnessMetrics
from ..metrics.relevance import RelevanceMetrics

class PostProcessor:
    """Post-processing validation and correction of generated answers"""
    
    def __init__(self):
        self.faithfulness = FaithfulnessMetrics()
        self.relevance = RelevanceMetrics()
        
    def validate_answer(self, query: str, context: str, answer: str, 
                       confidence_threshold: float = 0.7) -> Dict:
        """
        Validate generated answer and apply corrections if needed
        
        Args:
            query: Original query
            context: Source context
            answer: Generated answer
            confidence_threshold: Minimum confidence for direct answer
            
        Returns:
            Validation results with corrected answer if needed
        """
        # Calculate metrics
        faithfulness = self.faithfulness.faithfulness_score(answer, context)
        context_rel = self.relevance.context_relevance(query, context)
        answer_rel = self.relevance.answer_relevance(query, answer)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_confidence(
            faithfulness['faithfulness'],
            context_rel['semantic_relevance'],
            answer_rel['overall_relevance']
        )
        
        # Determine if correction is needed
        needs_correction = (
            faithfulness['faithfulness'] < 0.6 or
            overall_confidence < confidence_threshold or
            faithfulness['risk_level'] == 'HIGH'
        )
        
        # Apply corrections if needed
        if needs_correction:
            corrected_answer = self._apply_corrections(
                answer, context, query, overall_confidence
            )
            correction_applied = True
        else:
            corrected_answer = answer
            correction_applied = False
        
        return {
            'original_answer': answer,
            'corrected_answer': corrected_answer,
            'correction_applied': correction_applied,
            'confidence': overall_confidence,
            'faithfulness_score': faithfulness['faithfulness'],
            'context_relevance': context_rel['semantic_relevance'],
            'answer_relevance': answer_rel['overall_relevance'],
            'validation_passed': not needs_correction,
            'correction_reasons': self._get_correction_reasons(
                faithfulness, context_rel, answer_rel
            ) if needs_correction else []
        }
    
    def add_confidence_disclaimer(self, answer: str, confidence: float) -> str:
        """
        Add confidence-based disclaimer to answer
        
        Args:
            answer: Original answer
            confidence: Confidence score (0-1)
            
        Returns:
            Answer with appropriate disclaimer
        """
        if confidence >= 0.8:
            return answer  # No disclaimer for high confidence
        elif confidence >= 0.6:
            return f"Based on the available information: {answer}"
        elif confidence >= 0.4:
            return f"According to the context provided: {answer}"
        else:
            return "I don't have enough information to provide a confident answer."
    
    def extract_and_verify_claims(self, answer: str, context: str) -> Dict:
        """
        Extract claims from answer and verify against context
        
        Args:
            answer: Generated answer
            context: Source context
            
        Returns:
            Claim verification results
        """
        # Extract claims (simple sentence splitting)
        claims = re.split(r'[.!?]+', answer)
        claims = [c.strip() for c in claims if c.strip()]
        
        verified_claims = []
        unverified_claims = []
        
        for claim in claims:
            if len(claim.split()) < 3:  # Skip very short fragments
                continue
            
            # Simple verification: check if key terms are in context
            claim_words = set(claim.lower().split())
            context_words = set(context.lower().split())
            
            overlap = len(claim_words.intersection(context_words))
            total_claim_words = len(claim_words)
            
            if total_claim_words > 0:
                verification_score = overlap / total_claim_words
                is_verified = verification_score > 0.5
            else:
                verification_score = 0.0
                is_verified = False
            
            claim_info = {
                'claim': claim,
                'verification_score': verification_score,
                'is_verified': is_verified,
                'overlap_ratio': overlap / max(total_claim_words, 1)
            }
            
            if is_verified:
                verified_claims.append(claim_info)
            else:
                unverified_claims.append(claim_info)
        
        return {
            'verified_claims': verified_claims,
            'unverified_claims': unverified_claims,
            'verification_rate': len(verified_claims) / max(len(claims), 1),
            'total_claims': len(claims)
        }
    
    def _calculate_confidence(self, faithfulness: float, 
                            context_rel: float, answer_rel: float) -> float:
        """Calculate overall confidence score"""
        weights = {
            'faithfulness': 0.5,
            'context_relevance': 0.3,
            'answer_relevance': 0.2
        }
        
        confidence = (
            weights['faithfulness'] * faithfulness +
            weights['context_relevance'] * context_rel +
            weights['answer_relevance'] * answer_rel
        )
        
        return float(confidence)
    
    def _apply_corrections(self, answer: str, context: str, 
                          query: str, confidence: float) -> str:
        """
        Apply appropriate corrections based on confidence level
        
        Args:
            answer: Original answer
            context: Source context
            query: Original query
            confidence: Confidence score
            
        Returns:
            Corrected answer
        """
        if confidence < 0.3:
            # Very low confidence - provide safe response
            return "I cannot provide a confident answer based on the available information."
        
        elif confidence < 0.6:
            # Medium confidence - add disclaimer and be more conservative
            disclaimer = "Based on the limited information available: "
            
            # Try to extract only verified parts
            claim_verification = self.extract_and_verify_claims(answer, context)
            verified_claims = [c['claim'] for c in claim_verification['verified_claims']]
            
            if verified_claims:
                verified_answer = " ".join(verified_claims)
                return disclaimer + verified_answer
            else:
                return disclaimer + "The available context doesn't contain specific information to answer this question."
        
        else:
            # Reasonable confidence but needs minor adjustments
            # Remove unverified claims
            claim_verification = self.extract_and_verify_claims(answer, context)
            verified_claims = [c['claim'] for c in claim_verification['verified_claims']]
            
            if verified_claims:
                corrected = " ".join(verified_claims)
                # Add confidence disclaimer
                return self.add_confidence_disclaimer(corrected, confidence)
            else:
                return self.add_confidence_disclaimer(answer, confidence)
    
    def _get_correction_reasons(self, faithfulness: Dict, 
                               context_rel: Dict, answer_rel: Dict) -> List[str]:
        """Get reasons why correction was applied"""
        reasons = []
        
        if faithfulness['faithfulness'] < 0.6:
            reasons.append(f"Low faithfulness ({faithfulness['faithfulness']:.1%})")
        
        if context_rel['semantic_relevance'] < 0.5:
            reasons.append(f"Low context relevance ({context_rel['semantic_relevance']:.1%})")
        
        if answer_rel['overall_relevance'] < 0.6:
            reasons.append(f"Low answer relevance ({answer_rel['overall_relevance']:.1%})")
        
        if faithfulness['risk_level'] == 'HIGH':
            reasons.append("High hallucination risk detected")
        
        return reasons
