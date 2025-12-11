import re
from typing import Dict, List, Tuple
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

class RuleBasedDetector:
    """Rule-based hallucination detection using linguistic patterns"""
    
    def __init__(self):
        self.hallucination_patterns = [
            # Overly confident without evidence
            r'\b(undoubtedly|definitely|certainly|absolutely)\b',
            # Vague quantifiers
            r'\b(many|some|several|few|most|all)\b',
            # Speculative language
            r'\b(probably|likely|perhaps|maybe|possibly)\b',
            # Unsupported claims
            r'\b(it is known|studies show|research indicates)\b',
            # Contradiction indicators
            r'\b(however|but|although|despite|nevertheless)\b.*\b(but|however)\b',
        ]
        
        self.confidence_indicators = [
            r'\b(I am confident|with high confidence|certainly)\b',
            r'\b(based on|according to|as per)\b',
            r'\b(as stated|as mentioned|as described)\b',
        ]
    
    def detect(self, answer: str, context: str = None) -> Dict:
        """
        Detect potential hallucinations using rule-based patterns
        
        Args:
            answer: Generated answer to check
            context: Optional context for verification
            
        Returns:
            Detection results with confidence scores
        """
        results = {
            'patterns_found': [],
            'confidence_indicators': [],
            'pattern_scores': {},
            'overall_score': 0.0,
            'risk_level': 'UNKNOWN'
        }
        
        # Check for hallucination patterns
        pattern_matches = []
        for pattern in self.hallucination_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            if matches:
                pattern_matches.extend(matches)
                results['patterns_found'].append({
                    'pattern': pattern,
                    'matches': matches,
                    'count': len(matches)
                })
        
        # Check for confidence indicators
        confidence_matches = []
        for indicator in self.confidence_indicators:
            matches = re.findall(indicator, answer, re.IGNORECASE)
            if matches:
                confidence_matches.extend(matches)
                results['confidence_indicators'].append({
                    'indicator': indicator,
                    'matches': matches,
                    'count': len(matches)
                })
        
        # Calculate pattern-based score
        total_words = len(word_tokenize(answer))
        hallucination_word_count = sum(len(match.split()) for match in pattern_matches)
        
        if total_words > 0:
            pattern_score = hallucination_word_count / total_words
        else:
            pattern_score = 0.0
        
        # Adjust score based on confidence indicators
        confidence_word_count = sum(len(match.split()) for match in confidence_matches)
        confidence_adjustment = min(confidence_word_count / max(total_words, 1), 0.3)
        
        final_score = max(0, pattern_score - confidence_adjustment)
        
        # Determine risk level
        if final_score < 0.1:
            risk = 'LOW'
        elif final_score < 0.25:
            risk = 'MEDIUM'
        else:
            risk = 'HIGH'
        
        results['pattern_scores'] = {
            'hallucination_word_ratio': pattern_score,
            'confidence_adjustment': confidence_adjustment,
            'final_pattern_score': final_score
        }
        results['overall_score'] = final_score
        results['risk_level'] = risk
        
        return results
    
    def check_context_alignment(self, claim: str, context: str) -> Tuple[bool, float]:
        """
        Check if a specific claim aligns with context using keyword matching
        
        Args:
            claim: Specific claim to check
            context: Context to verify against
            
        Returns:
            Tuple of (is_aligned, confidence_score)
        """
        claim_words = set(word_tokenize(claim.lower()))
        context_words = set(word_tokenize(context.lower()))
        
        # Calculate Jaccard similarity
        intersection = claim_words.intersection(context_words)
        union = claim_words.union(context_words)
        
        if len(union) == 0:
            return False, 0.0
        
        similarity = len(intersection) / len(union)
        
        # Consider aligned if similarity > 0.3
        is_aligned = similarity > 0.3
        
        return is_aligned, similarity
