# measure_hallucination/evaluation/benchmark_datasets.py
import json
import pandas as pd
from typing import List, Dict, Any
import datasets

class BenchmarkDatasets:
    """Standard datasets for hallucination evaluation"""
    
    def __init__(self):
        self.datasets = {
            'hotpot_qa': 'hotpot_qa',
            'squad': 'squad',
            'truthful_qa': 'truthful_qa',
            'fever': 'fever'
        }
    
    def load_dataset(self, dataset_name: str, split: str = 'validation') -> List[Dict]:
        """
        Load benchmark dataset
        
        Args:
            dataset_name: Name of dataset to load
            split: Dataset split (train/validation/test)
            
        Returns:
            List of examples
        """
        if dataset_name not in self.datasets:
            available = list(self.datasets.keys())
            raise ValueError(f"Dataset {dataset_name} not found. Available: {available}")
        
        try:
            dataset = datasets.load_dataset(
                self.datasets[dataset_name], 
                split=split
            )
            
            # Convert to list of dictionaries
            examples = []
            for example in dataset:
                examples.append(dict(example))
            
            return examples
            
        except Exception as e:
            raise Exception(f"Error loading dataset {dataset_name}: {str(e)}")
    
    def create_hallucination_test_set(self, size: int = 100) -> List[Dict]:
        """
        Create custom hallucination test set
        
        Args:
            size: Number of test examples
            
        Returns:
            List of test examples with ground truth
        """
        test_set = []
        
        # Example 1: Perfect answer
        test_set.append({
            'id': 'perfect_001',
            'query': 'What is machine learning?',
            'context': 'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.',
            'ground_truth_answer': 'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.',
            'expected_faithfulness': 1.0,
            'has_hallucination': False,
            'difficulty': 'easy'
        })
        
        # Example 2: Partial hallucination
        test_set.append({
            'id': 'partial_001',
            'query': 'Who invented Python?',
            'context': 'Python was created by Guido van Rossum and first released in 1991.',
            'ground_truth_answer': 'Python was created by Guido van Rossum.',
            'expected_faithfulness': 0.8,
            'has_hallucination': True,
            'difficulty': 'medium'
        })
        
        # Example 3: Complete hallucination
        test_set.append({
            'id': 'hallucination_001',
            'query': 'What is the capital of Mars?',
            'context': 'Mars is the fourth planet from the Sun and has no countries or capitals.',
            'ground_truth_answer': 'Mars does not have a capital as it has no countries.',
            'expected_faithfulness': 0.0,
            'has_hallucination': True,
            'difficulty': 'hard'
        })
        
        # Add more examples up to size
        # (In practice, you would load from a file or database)
        
        return test_set[:size]
    
    def save_results(self, results: List[Dict], filepath: str):
        """
        Save evaluation results to file
        
        Args:
            results: Evaluation results
            filepath: Path to save file
        """
        df = pd.DataFrame(results)
        
        if filepath.endswith('.csv'):
            df.to_csv(filepath, index=False)
        elif filepath.endswith('.json'):
            df.to_json(filepath, orient='records', indent=2)
        elif filepath.endswith('.parquet'):
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError("Unsupported file format. Use .csv, .json, or .parquet")
        
        print(f"Results saved to {filepath}")
    
    def calculate_metrics(self, predictions: List[Dict], 
                         ground_truth: List[Dict]) -> Dict:
        """
        Calculate evaluation metrics
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            
        Returns:
            Dictionary of metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        metrics = {
            'total_examples': len(predictions),
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'faithfulness_correlation': 0.0,
            'detailed_breakdown': {}
        }
        
        # Calculate confusion matrix
        tp = fp = tn = fn = 0
        faithfulness_diffs = []
        
        for pred, truth in zip(predictions, ground_truth):
            pred_hallucination = pred.get('has_hallucination', False)
            truth_hallucination = truth.get('has_hallucination', False)
            
            if pred_hallucination and truth_hallucination:
                tp += 1
            elif pred_hallucination and not truth_hallucination:
                fp += 1
            elif not pred_hallucination and truth_hallucination:
                fn += 1
            else:
                tn += 1
            
            # Calculate faithfulness correlation
            pred_faith = pred.get('faithfulness_score', 0.0)
            truth_faith = truth.get('expected_faithfulness', 0.0)
            faithfulness_diffs.append(abs(pred_faith - truth_faith))
        
        # Calculate metrics
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        avg_faithfulness_diff = sum(faithfulness_diffs) / len(faithfulness_diffs) if faithfulness_diffs else 0.0
        
        metrics.update({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'faithfulness_correlation': 1 - avg_faithfulness_diff,  # Inverse of average difference
            'confusion_matrix': {
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            },
            'avg_faithfulness_difference': avg_faithfulness_diff
        })
        
        return metrics
