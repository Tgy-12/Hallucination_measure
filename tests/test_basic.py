# measure_hallucination/tests/test_basic.py
"""
Basic tests for the hallucination measurement toolkit
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measure_hallucination.metrics.faithfulness import FaithfulnessMetrics
from measure_hallucination.metrics.relevance import RelevanceMetrics
from measure_hallucination.detectors.rule_based import RuleBasedDetector

class TestHallucinationMetrics(unittest.TestCase):
    
    def setUp(self):
        self.query = "What is machine learning?"
        self.context = "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
        self.good_answer = "Machine learning is a subset of AI that allows systems to learn from experience."
        self.bad_answer = "Machine learning was invented in 1950 by Alan Turing and is primarily used for robotics."
        
    def test_faithfulness_good_answer(self):
        """Test faithfulness metrics on a good answer"""
        faithfulness = FaithfulnessMetrics()
        results = faithfulness.faithfulness_score(self.good_answer, self.context)
        
        self.assertGreaterEqual(results['faithfulness'], 0.7)
        self.assertEqual(results['risk_level'], 'LOW')
        self.assertGreater(results['supported_claims'], 0)
        
    def test_faithfulness_bad_answer(self):
        """Test faithfulness metrics on a bad answer"""
        faithfulness = FaithfulnessMetrics()
        results = faithfulness.faithfulness_score(self.bad_answer, self.context)
        
        self.assertLessEqual(results['faithfulness'], 0.3)
        self.assertEqual(results['risk_level'], 'HIGH')
        
    def test_relevance_calculation(self):
        """Test relevance metrics calculation"""
        relevance = RelevanceMetrics()
        context_rel = relevance.context_relevance(self.query, self.context)
        answer_rel = relevance.answer_relevance(self.query, self.good_answer)
        
        self.assertGreaterEqual(context_rel['semantic_relevance'], 0.7)
        self.assertGreaterEqual(answer_rel['overall_relevance'], 0.6)
        
    def test_rule_based_detection(self):
        """Test rule-based hallucination detection"""
        detector = RuleBasedDetector()
        
        # Test with confident but unverified statement
        test_answer = "Undoubtedly, machine learning was invented in 1956 at Dartmouth College."
        results = detector.detect(test_answer, self.context)
        
        self.assertGreater(results['overall_score'], 0.1)
        self.assertIn(results['risk_level'], ['LOW', 'MEDIUM', 'HIGH'])
        
    def test_claim_extraction(self):
        """Test claim extraction functionality"""
        faithfulness = FaithfulnessMetrics()
        claims = faithfulness.extract_claims(self.good_answer)
        
        self.assertGreaterEqual(len(claims), 1)
        self.assertIsInstance(claims, list)
        
    def test_confidence_scoring(self):
        """Test confidence scoring in relevance metrics"""
        relevance = RelevanceMetrics()
        context_rel = relevance.context_relevance(self.query, self.context)
        
        self.assertIn(context_rel['relevance_level'], ['LOW', 'MEDIUM', 'HIGH'])
        self.assertGreaterEqual(context_rel['semantic_relevance'], 0.0)
        self.assertLessEqual(context_rel['semantic_relevance'], 1.0)

if __name__ == '__main__':
    unittest.main()
