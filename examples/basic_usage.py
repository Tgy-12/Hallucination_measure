"""
Basic usage example for the Hallucination Measurement Toolkit
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measure_hallucination import (
    FaithfulnessMetrics,
    RelevanceMetrics,
    HybridDetector,
    PromptEngineer,
    PostProcessor
)

def main():
    print("🧠 Hallucination Measurement Toolkit - Basic Usage Example")
    print("=" * 60)
    
    # Example data
    query = "What is neurostack?"
    context = "Neurostack is a Retrieval-Augmented Generation (RAG) system developed for accurate question answering. It uses hybrid retrieval combining BM25 and FAISS for optimal context retrieval."
    answer = "Neurostack is a RAG system that uses both BM25 and FAISS for hybrid retrieval to provide accurate answers."
    
    print(f"Query: {query}")
    print(f"Context: {context[:100]}...")
    print(f"Answer: {answer}")
    print()
    
    # 1. Calculate Faithfulness
    print("📊 1. Faithfulness Analysis:")
    faithfulness = FaithfulnessMetrics()
    faith_results = faithfulness.faithfulness_score(answer, context)
    print(f"   Faithfulness Score: {faith_results['faithfulness']:.1%}")
    print(f"   Supported Claims: {faith_results['supported_claims']}/{faith_results['total_claims']}")
    print(f"   Risk Level: {faith_results['risk_level']}")
    print()
    
    # 2. Calculate Relevance
    print("🎯 2. Relevance Analysis:")
    relevance = RelevanceMetrics()
    context_rel = relevance.context_relevance(query, context)
    answer_rel = relevance.answer_relevance(query, answer)
    print(f"   Context Relevance: {context_rel['semantic_relevance']:.1%}")
    print(f"   Answer Relevance: {answer_rel['overall_relevance']:.1%}")
    print()
    
    # 3. Detect Hallucination
    print("🔍 3. Hallucination Detection:")
    detector = HybridDetector()
    detection = detector.detect(query, context, answer)
    print(f"   Decision: {detection['final_decision']['decision']}")
    print(f"   Confidence: {detection['final_decision']['confidence']:.1%}")
    print(f"   Severity: {detection['final_decision']['severity']}")
    print()
    
    # 4. Generate Anti-Hallucination Prompt
    print("🛡️ 4. Anti-Hallucination Prompt:")
    prompt_engineer = PromptEngineer()
    prompt = prompt_engineer.get_prompt(
        template_name='strict_context_only',
        context=context,
        question=query
    )
    print(f"   Prompt Template: strict_context_only")
    print(f"   Prompt Length: {len(prompt)} characters")
    print(f"   First 200 chars: {prompt[:200]}...")
    print()
    
    # 5. Post-processing Validation
    print("✅ 5. Post-processing Validation:")
    post_processor = PostProcessor()
    validation = post_processor.validate_answer(query, context, answer)
    print(f"   Validation Passed: {validation['validation_passed']}")
    print(f"   Overall Confidence: {validation['confidence']:.1%}")
    print(f"   Correction Applied: {validation['correction_applied']}")
    
    if validation['correction_applied']:
        print(f"   Corrected Answer: {validation['corrected_answer'][:100]}...")
    
    print()
    print("=" * 60)
    print("Example completed successfully! ✅")

if __name__ == "__main__":
    main()
