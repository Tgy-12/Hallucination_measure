# measure_hallucination/evaluation/evaluation_suite.py
import time
from typing import Dict, List, Any, Optional
import pandas as pd
from tqdm import tqdm
from ..metrics.composite import CompositeMetrics
from ..detectors.hybrid_detector import HybridDetector
from .benchmark_datasets import BenchmarkDatasets

class EvaluationSuite:
    """Comprehensive evaluation suite for hallucination detection"""
    
    def __init__(self):
        self.metrics = CompositeMetrics()
        self.detector = HybridDetector()
        self.datasets = BenchmarkDatasets()
        
    def evaluate_model(self, model_func, dataset_name: str = 'custom', 
                      num_samples: int = 50) -> Dict:
        """
        Evaluate a model on hallucination detection
        
        Args:
            model_func: Function that takes (query, context) and returns answer
            dataset_name: Name of dataset to use
            num_samples: Number of samples to evaluate
            
        Returns:
            Evaluation results
        """
        print(f"Evaluating model on {dataset_name} dataset...")
        
        # Load dataset
        if dataset_name == 'custom':
            dataset = self.datasets.create_hallucination_test_set(num_samples)
        else:
            dataset = self.datasets.load_dataset(dataset_name)
            dataset = dataset[:num_samples]  # Limit samples
        
        results = []
        inference_times = []
        
        # Evaluate each example
        for example in tqdm(dataset, desc="Evaluating"):
            query = example.get('query', '')
            context = example.get('context', '')
            ground_truth = example.get('ground_truth_answer', '')
            
            if not query or not context:
                continue
            
            # Time the model inference
            start_time = time.time()
            answer = model_func(query, context)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Calculate metrics
            metrics_result = self.metrics.calculate_all_metrics(
                query=query,
                context=context,
                answer=answer
            )
            
            # Detect hallucinations
            detection_result = self.detector.detect(
                query=query,
                context=context,
                answer=answer
            )
            
            # Store results
            example_result = {
                'id': example.get('id', ''),
                'query': query,
                'context_length': len(context),
                'answer_length': len(answer),
                'inference_time': inference_time,
                'faithfulness_score': metrics_result['faithfulness']['faithfulness'],
                'context_relevance': metrics_result['context_relevance']['semantic_relevance'],
                'answer_relevance': metrics_result['answer_relevance']['overall_relevance'],
                'hallucination_score': metrics_result['hallucination_score'],
                'risk_assessment': metrics_result['risk_assessment']['level'],
                'detection_decision': detection_result['final_decision']['decision'],
                'detection_confidence': detection_result['final_decision']['confidence'],
                'has_hallucination': metrics_result['hallucination_score'] > 0.5,
                'ground_truth_answer': ground_truth,
                'model_answer': answer,
                'recommendations': metrics_result['recommendations']
            }
            
            results.append(example_result)
        
        # Calculate summary statistics
        summary = self._calculate_summary(results, inference_times)
        
        return {
            'detailed_results': results,
            'summary': summary,
            'dataset_info': {
                'name': dataset_name,
                'size': len(results),
                'avg_context_length': pd.Series([r['context_length'] for r in results]).mean(),
                'avg_answer_length': pd.Series([r['answer_length'] for r in results]).mean()
            }
        }
    
    def compare_models(self, models: Dict[str, Any], dataset_name: str = 'custom',
                      num_samples: int = 30) -> Dict:
        """
        Compare multiple models on hallucination metrics
        
        Args:
            models: Dictionary of model_name: model_function pairs
            dataset_name: Dataset to use
            num_samples: Number of samples per model
            
        Returns:
            Comparison results
        """
        print(f"Comparing {len(models)} models on {dataset_name} dataset...")
        
        comparison_results = {}
        
        for model_name, model_func in models.items():
            print(f"\nEvaluating {model_name}...")
            
            results = self.evaluate_model(
                model_func=model_func,
                dataset_name=dataset_name,
                num_samples=num_samples
            )
            
            comparison_results[model_name] = {
                'summary': results['summary'],
                'dataset_info': results['dataset_info']
            }
        
        # Generate comparison summary
        comparison_summary = self._generate_comparison_summary(comparison_results)
        
        return {
            'model_results': comparison_results,
            'comparison_summary': comparison_summary,
            'best_model': max(comparison_results.items(), 
                            key=lambda x: x[1]['summary']['overall_score'])[0]
        }
    
    def generate_report(self, results: Dict, output_format: str = 'markdown') -> str:
        """
        Generate evaluation report
        
        Args:
            results: Evaluation results
            output_format: Format of report (markdown, html, json)
            
        Returns:
            Formatted report
        """
        if output_format == 'markdown':
            return self._generate_markdown_report(results)
        elif output_format == 'html':
            return self._generate_html_report(results)
        elif output_format == 'json':
            import json
            return json.dumps(results, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _calculate_summary(self, results: List[Dict], inference_times: List[float]) -> Dict:
        """Calculate summary statistics from results"""
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        
        summary = {
            'total_examples': len(results),
            'avg_inference_time': pd.Series(inference_times).mean(),
            'avg_faithfulness': df['faithfulness_score'].mean(),
            'avg_context_relevance': df['context_relevance'].mean(),
            'avg_answer_relevance': df['answer_relevance'].mean(),
            'avg_hallucination_score': df['hallucination_score'].mean(),
            'hallucination_rate': (df['hallucination_score'] > 0.5).mean(),
            'detection_accuracy': self._calculate_detection_accuracy(df),
            'risk_distribution': df['risk_assessment'].value_counts().to_dict(),
            'overall_score': self._calculate_overall_score(df)
        }
        
        return summary
    
    def _calculate_detection_accuracy(self, df: pd.DataFrame) -> float:
        """Calculate hallucination detection accuracy"""
        if 'has_hallucination' not in df.columns or 'detection_decision' not in df.columns:
            return 0.0
        
        # Map decisions to binary predictions
        def decision_to_binary(decision):
            if decision == 'HALLUCINATION':
                return True
            elif decision == 'VALID':
                return False
            else:
                return None
        
        df['predicted_hallucination'] = df['detection_decision'].apply(decision_to_binary)
        
        # Calculate accuracy
        valid_predictions = df.dropna(subset=['predicted_hallucination'])
        if len(valid_predictions) == 0:
            return 0.0
        
        accuracy = (valid_predictions['has_hallucination'] == 
                   valid_predictions['predicted_hallucination']).mean()
        
        return float(accuracy)
    
    def _calculate_overall_score(self, df: pd.DataFrame) -> float:
        """Calculate overall performance score"""
        weights = {
            'faithfulness': 0.4,
            'detection_accuracy': 0.3,
            'inverse_hallucination_rate': 0.2,
            'answer_relevance': 0.1
        }
        
        # Get average values
        avg_faithfulness = df['faithfulness_score'].mean()
        detection_accuracy = self._calculate_detection_accuracy(df)
        hallucination_rate = (df['hallucination_score'] > 0.5).mean()
        avg_answer_relevance = df['answer_relevance'].mean()
        
        # Calculate weighted score
        overall_score = (
            weights['faithfulness'] * avg_faithfulness +
            weights['detection_accuracy'] * detection_accuracy +
            weights['inverse_hallucination_rate'] * (1 - hallucination_rate) +
            weights['answer_relevance'] * avg_answer_relevance
        )
        
        return float(overall_score)
    
    def _generate_comparison_summary(self, comparison_results: Dict) -> Dict:
        """Generate comparison summary across models"""
        summary = {}
        
        for model_name, results in comparison_results.items():
            model_summary = results['summary']
            summary[model_name] = {
                'overall_score': model_summary.get('overall_score', 0),
                'avg_faithfulness': model_summary.get('avg_faithfulness', 0),
                'hallucination_rate': model_summary.get('hallucination_rate', 0),
                'detection_accuracy': model_summary.get('detection_accuracy', 0),
                'avg_inference_time': model_summary.get('avg_inference_time', 0)
            }
        
        return summary
    
    def _generate_markdown_report(self, results: Dict) -> str:
        """Generate markdown report"""
        report = []
        
        # Header
        report.append("# Hallucination Evaluation Report")
        report.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary section
        if 'summary' in results:
            summary = results['summary']
            report.append("## 📊 Summary Statistics")
            report.append("")
            report.append("| Metric | Value |")
            report.append("|--------|-------|")
            report.append(f"| Total Examples | {summary.get('total_examples', 0)} |")
            report.append(f"| Average Inference Time | {summary.get('avg_inference_time', 0):.3f}s |")
            report.append(f"| Average Faithfulness | {summary.get('avg_faithfulness', 0):.1%} |")
            report.append(f"| Average Context Relevance | {summary.get('avg_context_relevance', 0):.1%} |")
            report.append(f"| Average Answer Relevance | {summary.get('avg_answer_relevance', 0):.1%} |")
            report.append(f"| Average Hallucination Score | {summary.get('avg_hallucination_score', 0):.3f} |")
            report.append(f"| Hallucination Rate | {summary.get('hallucination_rate', 0):.1%} |")
            report.append(f"| Detection Accuracy | {summary.get('detection_accuracy', 0):.1%} |")
            report.append(f"| Overall Score | {summary.get('overall_score', 0):.3f} |")
            report.append("")
        
        # Risk distribution
        if 'summary' in results and 'risk_distribution' in results['summary']:
            report.append("## ⚠️ Risk Distribution")
            report.append("")
            for risk, count in results['summary']['risk_distribution'].items():
                report.append(f"- **{risk}**: {count} examples")
            report.append("")
        
        # Recommendations
        if 'detailed_results' in results and results['detailed_results']:
            report.append("## 🎯 Top Recommendations")
            report.append("")
            
            # Get unique recommendations
            all_recommendations = []
            for result in results['detailed_results']:
                if 'recommendations' in result:
                    all_recommendations.extend(result['recommendations'])
            
            # Count frequency
            from collections import Counter
            rec_counts = Counter(all_recommendations)
            
            for rec, count in rec_counts.most_common(5):
                report.append(f"1. {rec} ({count} occurrences)")
            report.append("")
        
        return "\n".join(report)
    
    def _generate_html_report(self, results: Dict) -> str:
        """Generate HTML report (simplified version)"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hallucination Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .good { color: green; }
                .warning { color: orange; }
                .bad { color: red; }
            </style>
        </head>
        <body>
            <h1>Hallucination Evaluation Report</h1>
            <p>Generated: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        """
        
        if 'summary' in results:
            summary = results['summary']
            html += """
            <h2>Summary Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
            """
            
            # Helper function to determine status
            def get_status(value, metric):
                if metric == 'hallucination_rate':
                    if value < 0.1: return '<span class="good">✓ Good</span>'
                    elif value < 0.3: return '<span class="warning">⚠ Warning</span>'
                    else: return '<span class="bad">✗ Poor</span>'
                elif metric == 'avg_faithfulness':
                    if value > 0.8: return '<span class="good">✓ Good</span>'
                    elif value > 0.6: return '<span class="warning">⚠ Warning</span>'
                    else: return '<span class="bad">✗ Poor</span>'
                else:
                    return ''
            
            metrics = [
                ('Total Examples', summary.get('total_examples', 0), ''),
                ('Average Inference Time', f"{summary.get('avg_inference_time', 0):.3f}s", ''),
                ('Average Faithfulness', f"{summary.get('avg_faithfulness', 0):.1%}", 
                 get_status(summary.get('avg_faithfulness', 0), 'avg_faithfulness')),
                ('Hallucination Rate', f"{summary.get('hallucination_rate', 0):.1%}", 
                 get_status(summary.get('hallucination_rate', 0), 'hallucination_rate')),
                ('Detection Accuracy', f"{summary.get('detection_accuracy', 0):.1%}", ''),
                ('Overall Score', f"{summary.get('overall_score', 0):.3f}", '')
            ]
            
            for name, value, status in metrics:
                html += f"<tr><td>{name}</td><td>{value}</td><td>{status}</td></tr>"
            
            html += "</table>"
        
        html += """
        </body>
        </html>
        """
        
        return html


