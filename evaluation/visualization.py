# measure_hallucination/evaluation/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualization:
    """Visualization tools for hallucination evaluation results"""
    
    def __init__(self, style: str = 'seaborn'):
        if style == 'seaborn':
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (12, 8)
        elif style == 'plotly':
            # Plotly will be used interactively
            pass
    
    def plot_metrics_distribution(self, results: List[Dict], 
                                 save_path: Optional[str] = None):
        """
        Plot distribution of key metrics
        
        Args:
            results: List of evaluation results
            save_path: Optional path to save figure
        """
        df = pd.DataFrame(results)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Faithfulness distribution
        axes[0, 0].hist(df['faithfulness_score'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].axvline(df['faithfulness_score'].mean(), color='red', 
                          linestyle='dashed', linewidth=2)
        axes[0, 0].set_title('Faithfulness Score Distribution')
        axes[0, 0].set_xlabel('Faithfulness Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].text(0.05, 0.95, f'Mean: {df["faithfulness_score"].mean():.3f}',
                       transform=axes[0, 0].transAxes, verticalalignment='top')
        
        # 2. Hallucination score distribution
        axes[0, 1].hist(df['hallucination_score'], bins=20, alpha=0.7, color='salmon')
        axes[0, 1].axvline(0.5, color='red', linestyle='dashed', linewidth=2, 
                          label='Threshold (0.5)')
        axes[0, 1].set_title('Hallucination Score Distribution')
        axes[0, 1].set_xlabel('Hallucination Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # 3. Context relevance distribution
        axes[1, 0].hist(df['context_relevance'], bins=20, alpha=0.7, color='lightgreen')
        axes[1, 0].axvline(df['context_relevance'].mean(), color='red', 
                          linestyle='dashed', linewidth=2)
        axes[1, 0].set_title('Context Relevance Distribution')
        axes[1, 0].set_xlabel('Context Relevance')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Risk level distribution
        risk_counts = df['risk_assessment'].value_counts()
        axes[1, 1].bar(risk_counts.index, risk_counts.values, 
                       color=['green', 'yellow', 'orange', 'red'])
        axes[1, 1].set_title('Risk Level Distribution')
        axes[1, 1].set_xlabel('Risk Level')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, results: List[Dict], 
                               save_path: Optional[str] = None):
        """
        Plot correlation matrix between metrics
        
        Args:
            results: List of evaluation results
            save_path: Optional path to save figure
        """
        df = pd.DataFrame(results)
        
        # Select numeric columns for correlation
        numeric_cols = ['faithfulness_score', 'context_relevance', 
                       'answer_relevance', 'hallucination_score', 
                       'detection_confidence', 'inference_time']
        
        # Filter columns that exist
        existing_cols = [col for col in numeric_cols if col in df.columns]
        correlation_df = df[existing_cols]
        
        # Calculate correlation matrix
        corr_matrix = correlation_df.corr()
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Evaluation Metrics')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved to {save_path}")
        
        plt.show()
    
    def plot_interactive_dashboard(self, results: List[Dict]):
        """
        Create interactive Plotly dashboard
        
        Args:
            results: List of evaluation results
        """
        df = pd.DataFrame(results)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Faithfulness vs Hallucination Score',
                           'Context Relevance Distribution',
                           'Risk Level Breakdown',
                           'Inference Time Distribution'),
            specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
                  [{'type': 'pie'}, {'type': 'box'}]]
        )
        
        # 1. Scatter plot: Faithfulness vs Hallucination
        fig.add_trace(
            go.Scatter(
                x=df['faithfulness_score'],
                y=df['hallucination_score'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['context_relevance'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Context Relevance")
                ),
                text=df['query'].str[:50] + '...',  # Show first 50 chars of query
                hoverinfo='text+x+y'
            ),
            row=1, col=1
        )
        
        # 2. Histogram: Context Relevance
        fig.add_trace(
            go.Histogram(
                x=df['context_relevance'],
                nbinsx=20,
                marker_color='lightblue',
                name='Context Relevance'
            ),
            row=1, col=2
        )
        
        # 3. Pie chart: Risk Level Breakdown
        risk_counts = df['risk_assessment'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.3,
                marker_colors=['green', 'yellow', 'orange', 'red']
            ),
            row=2, col=1
        )
        
        # 4. Box plot: Inference Time
        fig.add_trace(
            go.Box(
                y=df['inference_time'],
                name='Inference Time',
                boxpoints='outliers',
                marker_color='lightcoral'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Hallucination Evaluation Dashboard",
            height=800,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Faithfulness Score", row=1, col=1)
        fig.update_yaxes(title_text="Hallucination Score", row=1, col=1)
        fig.update_xaxes(title_text="Context Relevance", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Inference Time (seconds)", row=2, col=2)
        
        fig.show()
    
    def create_performance_report(self, results: List[Dict], 
                                 model_name: str = "Model") -> str:
        """
        Create comprehensive performance report
        
        Args:
            results: Evaluation results
            model_name: Name of the model being evaluated
            
        Returns:
            HTML report string
        """
        df = pd.DataFrame(results)
        
        # Calculate key metrics
        avg_faithfulness = df['faithfulness_score'].mean()
        avg_hallucination = df['hallucination_score'].mean()
        hallucination_rate = (df['hallucination_score'] > 0.5).mean()
        avg_inference_time = df['inference_time'].mean()
        
        # Risk distribution
        risk_dist = df['risk_assessment'].value_counts()
        
        # Create HTML report
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{model_name} - Hallucination Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .metric-card {{ background: white; border-radius: 10px; padding: 20px; 
                              margin: 15px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 2em; font-weight: bold; margin: 10px 0; }}
                .good {{ color: #10b981; }}
                .warning {{ color: #f59e0b; }}
                .poor {{ color: #ef4444; }}
                .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                          gap: 20px; margin: 30px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f8fafc; }}
                .recommendation {{ background: #f0f9ff; border-left: 4px solid #3b82f6; 
                                 padding: 15px; margin: 15px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 {model_name} - Hallucination Performance Report</h1>
                <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Examples Evaluated: {len(df)}</p>
            </div>
            
            <div class="summary">
                <div class="metric-card">
                    <h3>Average Faithfulness</h3>
                    <div class="metric-value {'good' if avg_faithfulness > 0.8 else 'warning' if avg_faithfulness > 0.6 else 'poor'}">
                        {(avg_faithfulness * 100):.1f}%
                    </div>
                    <p>Percentage of claims supported by context</p>
                </div>
                
                <div class="metric-card">
                    <h3>Hallucination Rate</h3>
                    <div class="metric-value {'good' if hallucination_rate < 0.1 else 'warning' if hallucination_rate < 0.3 else 'poor'}">
                        {(hallucination_rate * 100):.1f}%
                    </div>
                    <p>Percentage of answers with significant hallucination</p>
                </div>
                
                <div class="metric-card">
                    <h3>Average Inference Time</h3>
                    <div class="metric-value">
                        {avg_inference_time:.3f}s
                    </div>
                    <p>Average time to generate and validate answers</p>
                </div>
                
                <div class="metric-card">
                    <h3>Average Hallucination Score</h3>
                    <div class="metric-value {'good' if avg_hallucination < 0.2 else 'warning' if avg_hallucination < 0.4 else 'poor'}">
                        {avg_hallucination:.3f}
                    </div>
                    <p>Lower is better (0 = no hallucination, 1 = complete hallucination)</p>
                </div>
            </div>
            
            <h2>📊 Risk Level Distribution</h2>
            <table>
                <tr>
                    <th>Risk Level</th>
                    <th>Count</th>
                    <th>Percentage</th>
                    <th>Description</th>
                </tr>
        """
        
        # Add risk distribution rows
        for risk_level, count in risk_dist.items():
            percentage = (count / len(df)) * 100
            description = {
                'LOW': 'Minimal hallucination risk',
                'MEDIUM': 'Moderate risk, needs monitoring',
                'HIGH': 'High risk, requires attention',
                'VERY_HIGH': 'Critical risk, immediate action needed'
            }.get(risk_level, 'Unknown risk level')
            
            html_report += f"""
                <tr>
                    <td><strong>{risk_level}</strong></td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                    <td>{description}</td>
                </tr>
            """
        
        html_report += """
            </table>
            
            <h2>🎯 Top Recommendations</h2>
        """
        
        # Generate recommendations based on metrics
        recommendations = []
        
        if avg_faithfulness < 0.7:
            recommendations.append("Improve context grounding in generated answers")
        
        if hallucination_rate > 0.2:
            recommendations.append("Increase confidence thresholds for answer generation")
        
        if avg_inference_time > 2.0:
            recommendations.append("Optimize retrieval and validation pipelines")
        
        if len(df[df['context_relevance'] < 0.5]) > len(df) * 0.3:
            recommendations.append("Enhance context retrieval relevance")
        
        # Add default recommendation if none
        if not recommendations:
            recommendations.append("All metrics within acceptable ranges. Maintain current configuration.")
        
        # Add recommendations to report
        for i, rec in enumerate(recommendations[:5], 1):
            html_report += f"""
            <div class="recommendation">
                <h4>Recommendation #{i}</h4>
                <p>{rec}</p>
            </div>
            """
        
        html_report += """
            <h2>📈 Performance Trends</h2>
            <p><em>Note: For detailed visualizations, run the visualization tools provided in the toolkit.</em></p>
            
            <div style="margin-top: 40px; padding: 20px; background: #f8fafc; border-radius: 10px;">
                <h3>📋 Evaluation Summary</h3>
                <p>This report provides a comprehensive analysis of hallucination performance. 
                Key metrics to monitor:</p>
                <ul>
                    <li><strong>Faithfulness Score:</strong> Should be above 80% for production systems</li>
                    <li><strong>Hallucination Rate:</strong> Should be below 10% for critical applications</li>
                    <li><strong>Context Relevance:</strong> Should be above 70% for accurate answers</li>
                    <li><strong>Risk Distribution:</strong> Majority should be in LOW/MEDIUM categories</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html_report
