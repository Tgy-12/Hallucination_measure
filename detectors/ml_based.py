import numpy as np
from typing import Dict
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

class MLBasedDetector:
    """Machine learning based hallucination detection"""
    
    def __init__(self, model_path: str = None):
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf = TfidfVectorizer(max_features=1000)
        
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.is_trained = True
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.is_trained = False
            self.model_path = model_path
    
    def extract_features(self, query: str, context: str, answer: str) -> np.ndarray:
        """
        Extract features for ML model
        
        Args:
            query: User query
            context: Retrieved context
            answer: Generated answer
            
        Returns:
            Feature vector
        """
        features = []
        
        # 1. Semantic similarity features
        query_emb = self.sbert_model.encode(query)
        context_emb = self.sbert_model.encode(context)
        answer_emb = self.sbert_model.encode(answer)
        
        # Cosine similarities
        features.append(self._cosine_sim(query_emb, context_emb))
        features.append(self._cosine_sim(query_emb, answer_emb))
        features.append(self._cosine_sim(context_emb, answer_emb))
        
        # 2. Length-based features
        features.append(len(query.split()) / 100)  # Normalized
        features.append(len(context.split()) / 500)
        features.append(len(answer.split()) / 200)
        
        # 3. Overlap features
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())
        
        # Jaccard similarities
        features.append(self._jaccard_sim(query_words, context_words))
        features.append(self._jaccard_sim(query_words, answer_words))
        features.append(self._jaccard_sim(context_words, answer_words))
        
        # 4. TF-IDF features (first 10)
        try:
            tfidf_features = self._get_tfidf_features(query, context, answer)
            features.extend(tfidf_features[:10])  # Take first 10 features
        except:
            features.extend([0] * 10)  # Padding if TF-IDF fails
        
        return np.array(features).reshape(1, -1)
    
    def detect(self, query: str, context: str, answer: str) -> Dict:
        """
        Detect hallucination using ML model
        
        Args:
            query: User query
            context: Retrieved context
            answer: Generated answer
            
        Returns:
            Detection results
        """
        if not self.is_trained:
            return {
                'error': 'Model not trained',
                'suggestion': 'Train model first or provide pretrained model path'
            }
        
        # Extract features
        features = self.extract_features(query, context, answer)
        
        # Predict
        prediction = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0]
        
        # Get feature importance if available
        importance = {}
        if hasattr(self.model, 'feature_importances_'):
            top_indices = np.argsort(self.model.feature_importances_)[-5:]  # Top 5
            importance = {
                'top_features': [f'feature_{i}' for i in top_indices],
                'importance_scores': self.model.feature_importances_[top_indices].tolist()
            }
        
        return {
            'prediction': 'HALLUCINATION' if prediction == 1 else 'VALID',
            'confidence': float(max(proba)),
            'probabilities': {
                'valid': float(proba[0]),
                'hallucination': float(proba[1]) if len(proba) > 1 else 0.0
            },
            'feature_importance': importance,
            'num_features': features.shape[1]
        }
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the ML model"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Save model if path provided
        if hasattr(self, 'model_path') and self.model_path:
            joblib.dump(self.model, self.model_path)
        
        # Calculate training accuracy
        train_acc = self.model.score(X_train, y_train)
        
        results = {
            'training_accuracy': train_acc,
            'model_type': type(self.model).__name__,
            'is_trained': True
        }
        
        # Validation accuracy if provided
        if X_val is not None and y_val is not None:
            val_acc = self.model.score(X_val, y_val)
            results['validation_accuracy'] = val_acc
        
        return results
    
    def _cosine_sim(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _jaccard_sim(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity"""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _get_tfidf_features(self, query: str, context: str, answer: str) -> list:
        """Extract TF-IDF features"""
        # Combine all text
        combined = [query, context, answer]
        
        # Fit and transform
        tfidf_matrix = self.tfidf.fit_transform(combined)
        
        # Get average TF-IDF scores across documents
        avg_tfidf = tfidf_matrix.mean(axis=0).A1
        
        return avg_tfidf.tolist()
