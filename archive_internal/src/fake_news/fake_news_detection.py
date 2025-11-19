"""
Fake News Detection System

This module implements a machine learning pipeline for detecting fake news
using TF-IDF vectorization and PassiveAggressiveClassifier.

Author: Chaghoub Aymen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from typing import Tuple, List, Optional
from pathlib import Path

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class FakeNewsDetector:
    """
    A class for detecting fake news using machine learning techniques.
    
    This class handles data loading, preprocessing, model training, and evaluation
    for fake news detection.
    """
    
    def __init__(self, data_path: str = 'train.csv'):
        """
        Initialize the FakeNewsDetector.
        
        Args:
            data_path: Path to the training data CSV file
        """
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[PassiveAggressiveClassifier] = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and explore the dataset.
        
        Returns:
            Loaded DataFrame
        """
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            print(f"\n📋 First few rows:")
            print(self.df.head())
            print(f"\n❌ Missing values:")
            print(self.df.isnull().sum())
            return self.df
        except FileNotFoundError:
            print(f"❌ File {self.data_path} not found!")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text data.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove special characters, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', '', text)
        return text.strip()
    
    def prepare_data(self) -> Tuple[List[str], List[int]]:
        """
        Prepare the data for training by merging title and text, and cleaning.
        
        Returns:
            Tuple of (features, labels)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Merge title and text
        self.df['content'] = self.df['title'].fillna('') + " " + self.df['text'].fillna('')
        
        # Clean text
        print("🧹 Cleaning text data...")
        self.df['content'] = self.df['content'].apply(self.preprocess_text)
        
        # Download stopwords if not already downloaded
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        self.df['content'] = self.df['content'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in stop_words])
        )
        
        return self.df['content'].tolist(), self.df['label'].tolist()
    
    def train_model(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Train the fake news detection model.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        # Prepare data
        X, y = self.prepare_data()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Vectorize text
        print("🔢 Vectorizing text with TF-IDF...")
        self.vectorizer = TfidfVectorizer(max_df=0.7)
        self.X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        self.X_test_tfidf = self.vectorizer.transform(self.X_test)
        
        # Train model
        print("🤖 Training PassiveAggressiveClassifier...")
        self.model = PassiveAggressiveClassifier(max_iter=50)
        self.model.fit(self.X_train_tfidf, self.y_train)
        
        print("✅ Model training completed!")
    
    def evaluate_model(self) -> dict:
        """
        Evaluate the trained model and return metrics.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        
        print(f"\n📊 Model Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    def visualize_results(self, save_path: str = 'fake_news_confusion_matrix.png') -> None:
        """
        Create visualization of the confusion matrix.
        
        Args:
            save_path: Path to save the visualization
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        y_pred = self.model.predict(self.X_test_tfidf)
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                   xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
        plt.title("Confusion Matrix - Fake News Detection", fontsize=16, fontweight='bold')
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("Actual", fontsize=12)
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved: {save_path}")
        plt.show()
    
    def predict(self, text: str) -> Tuple[int, float]:
        """
        Predict if a given text is fake news.
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            cleaned_text = ' '.join([word for word in cleaned_text.split() 
                                   if word not in stop_words])
        except LookupError:
            pass  # Stopwords already removed during training
        
        # Vectorize and predict
        text_vectorized = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_vectorized)[0]
        
        # Get confidence score
        if hasattr(self.model, 'predict_proba'):
            confidence = self.model.predict_proba(text_vectorized)[0].max()
        else:
            confidence = 1.0  # PassiveAggressiveClassifier doesn't have predict_proba
        
        return prediction, confidence


def main():
    """
    Main function to run the fake news detection pipeline.
    """
    print("🗞️  Fake News Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = FakeNewsDetector('train.csv')
    
    try:
        # Load and explore data
        detector.load_data()
        
        # Train model
        detector.train_model()
        
        # Evaluate model
        metrics = detector.evaluate_model()
        
        # Create visualization
        detector.visualize_results()
        
        # Example prediction
        print("\n🧪 Example Prediction:")
        test_text = "This is a sample news article for testing."
        prediction, confidence = detector.predict(test_text)
        result = "FAKE" if prediction == 0 else "TRUE"
        print(f"Text: '{test_text}'")
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.2f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
