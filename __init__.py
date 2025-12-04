# measure_hallucination/__init__.py
"""
Hallucination Measurement and Mitigation Toolkit

A comprehensive toolkit for detecting, measuring, and mitigating hallucination 
in LLM and RAG systems. Part of the Neurostack-RAG-Copilot ecosystem.
"""

from .metrics.faithfulness import FaithfulnessMetrics
from .metrics.relevance import RelevanceMetrics
from .metrics.consistency import ConsistencyMetrics
from .metrics.composite import CompositeMetrics
from .detectors.rule_based import RuleBasedDetector
from .detectors.ml_based import MLBasedDetector
from .detectors.hybrid_detector import HybridDetector
from .mitigation.prompt_engineering import PromptEngineer
from .mitigation.retrieval_augmentation import RetrievalAugmenter
from .mitigation.post_processing import PostProcessor
from .evaluation.evaluation_suite import EvaluationSuite
from .evaluation.visualization import Visualization

__version__ = "1.0.0"
__author__ = "Tsegay Araya"
__email__ = ""
__description__ = "Comprehensive hallucination measurement and mitigation toolkit"
__url__ = "https://github.com/Tgy-12/Neurostack-RAG-Copilot/tree/main/measure_hallucination"

__all__ = [
    # Metrics
    "FaithfulnessMetrics",
    "RelevanceMetrics", 
    "ConsistencyMetrics",
    "CompositeMetrics",
    
    # Detectors
    "RuleBasedDetector",
    "MLBasedDetector", 
    "HybridDetector",
    
    # Mitigation
    "PromptEngineer",
    "RetrievalAugmenter",
    "PostProcessor",
    
    # Evaluation
    "EvaluationSuite",
    "Visualization",
]
