"""
Twitter Sentiment Analysis System

A comprehensive NLP system for analyzing sentiment in tweets using
multiple machine learning algorithms and advanced text preprocessing.

Author: Chaghoub Aymen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')

# NLP & Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class TwitterSentimentAnalyzer:
    """
    A comprehensive Twitter sentiment analysis system.
    
    This class handles data loading, text preprocessing, feature extraction,
    model training, and evaluation for sentiment analysis.
    """
    
    def __init__(self, data_path: str = 'tweets.csv'):
        """
        Initialize the TwitterSentimentAnalyzer.
        
        Args:
            data_path: Path to the tweets dataset CSV file
        """
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.models: Dict = {}
        self.results: Dict = {}
        self.best_model = None
        self.best_model_name = None
        
        # Configure plotting
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set2")
        
        # Download NLTK resources
        self._download_nltk_resources()
    
    def _download_nltk_resources(self) -> None:
        """
        Download required NLTK resources.
        """
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        
        print("✅ NLTK resources downloaded")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and explore the tweets dataset.
        
        Returns:
            Loaded DataFrame
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.data_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.df is None:
                raise ValueError("Could not read file with any encoding")
            
            print(f"✅ Dataset loaded: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        except FileNotFoundError:
            print(f"⚠️  File {self.data_path} not found. Creating sample data...")
            self._create_sample_data()
        
        print(f"\n📋 Dataset overview:")
        print(self.df.head(10))
        print(f"\n🔍 Data types:")
        print(self.df.info())
        
        # Identify sentiment and text columns
        sentiment_col = None
        text_col = None
        
        for col in self.df.columns:
            if 'sentiment' in col.lower():
                sentiment_col = col
            if 'text' in col.lower() or 'tweet' in col.lower():
                text_col = col
        
        if sentiment_col is None:
            sentiment_col = self.df.columns[0]
        if text_col is None:
            text_col = self.df.columns[1] if len(self.df.columns) > 1 else self.df.columns[0]
        
        print(f"\n✅ Sentiment column: '{sentiment_col}'")
        print(f"✅ Text column: '{text_col}'")
        
        # Rename columns for consistency
        self.df = self.df.rename(columns={sentiment_col: 'sentiment', text_col: 'text'})
        
        # Check missing values
        print(f"\n❌ Missing values:")
        print(self.df.isnull().sum())
        
        # Remove rows with missing critical data
        initial_size = len(self.df)
        self.df = self.df.dropna(subset=['text', 'sentiment'])
        print(f"✅ Removed {initial_size - len(self.df)} rows with missing data")
        
        # Remove duplicates
        duplicates = self.df.duplicated().sum()
        self.df = self.df.drop_duplicates()
        print(f"✅ Removed {duplicates} duplicate rows")
        
        # Sentiment distribution
        print(f"\n📊 Sentiment distribution:")
        sentiment_counts = self.df['sentiment'].value_counts()
        print(sentiment_counts)
        print(f"\nPercentages:")
        print(sentiment_counts / len(self.df) * 100)
        
        return self.df
    
    def _create_sample_data(self) -> None:
        """
        Create sample tweet data for demonstration.
        """
        sample_tweets = [
            'I love this airline! Great service and comfortable seats.',
            'Terrible experience, my flight was delayed for 5 hours.',
            'The flight was okay, nothing special but not bad either.',
            'Amazing staff and great customer service!',
            'Delayed flight, very disappointed with the service.',
            'Not bad but could be better, average experience.',
            'Excellent customer service and friendly staff!',
            'Worst airline ever, lost my luggage and no help.',
            'Average flight experience, nothing to complain about.',
            'Highly recommend this airline, great value for money!'
        ] * 100  # Repeat to create more data
        
        sample_sentiments = [
            'positive', 'negative', 'neutral', 'positive', 'negative',
            'neutral', 'positive', 'negative', 'neutral', 'positive'
        ] * 100
        
        self.df = pd.DataFrame({
            'text': sample_tweets,
            'sentiment': sample_sentiments
        })
        
        print(f"✅ Sample data created: {self.df.shape[0]} tweets")
    
    def clean_text(self, text: str) -> str:
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
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions (@user)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (#)
        text = re.sub(r'#\w+', '', text)
        
        # Remove special characters, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_text(self) -> pd.DataFrame:
        """
        Apply comprehensive text preprocessing.
        
        Returns:
            DataFrame with processed text
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n🧹 Preprocessing text data...")
        
        # Clean text
        print("🔧 Cleaning text...")
        self.df['clean_text'] = self.df['text'].apply(self.clean_text)
        
        # Show examples
        print("\n📝 Text cleaning examples:")
        for i in range(3):
            print(f"\nBefore: {self.df['text'].iloc[i][:80]}...")
            print(f"After:  {self.df['clean_text'].iloc[i][:80]}...")
        
        # Remove stopwords
        print("\n🔧 Removing stopwords...")
        stop_words = set(stopwords.words('english'))
        
        def remove_stopwords(text: str) -> str:
            words = text.split()
            return ' '.join([word for word in words if word not in stop_words and len(word) > 2])
        
        self.df['processed_text'] = self.df['clean_text'].apply(remove_stopwords)
        
        # Lemmatization
        print("🔧 Lemmatizing text...")
        lemmatizer = WordNetLemmatizer()
        
        def lemmatize_text(text: str) -> str:
            words = text.split()
            return ' '.join([lemmatizer.lemmatize(word) for word in words])
        
        self.df['final_text'] = self.df['processed_text'].apply(lemmatize_text)
        
        # Remove empty texts
        initial_size = len(self.df)
        self.df = self.df[self.df['final_text'].str.len() > 0]
        print(f"✅ Removed {initial_size - len(self.df)} empty texts after preprocessing")
        
        # Add text length features
        self.df['text_length'] = self.df['text'].astype(str).apply(len)
        self.df['word_count'] = self.df['text'].astype(str).apply(lambda x: len(x.split()))
        
        print(f"✅ Text preprocessing completed")
        print(f"✅ Final dataset: {len(self.df)} tweets")
        
        return self.df
    
    def create_visualizations(self, save_path: str = 'twitter_analysis.png') -> None:
        """
        Create comprehensive visualizations for the analysis.
        
        Args:
            save_path: Path to save the visualization
        """
        if self.df is None:
            raise ValueError("Data not processed. Complete preprocessing first.")
        
        print("\n📊 Creating visualizations...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Sentiment distribution
        ax1 = plt.subplot(2, 3, 1)
        sentiment_counts = self.df['sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#95a5a6'][:len(sentiment_counts)]
        sentiment_counts.plot(kind='bar', color=colors, edgecolor='black')
        plt.title('Sentiment Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Tweets')
        plt.xticks(rotation=45)
        
        for i, v in enumerate(sentiment_counts.values):
            plt.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 2. Tweet length distribution
        ax2 = plt.subplot(2, 3, 2)
        plt.hist(self.df['word_count'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(self.df['word_count'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {self.df["word_count"].mean():.1f}')
        plt.title('Tweet Length Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 3. Average length by sentiment
        ax3 = plt.subplot(2, 3, 3)
        avg_length = self.df.groupby('sentiment')['word_count'].mean().sort_values()
        avg_length.plot(kind='barh', color='coral', edgecolor='black')
        plt.title('Average Tweet Length by Sentiment', fontsize=14, fontweight='bold')
        plt.xlabel('Average Number of Words')
        plt.ylabel('Sentiment')
        
        # 4. Most frequent words
        ax4 = plt.subplot(2, 3, 4)
        all_words = ' '.join(self.df['final_text']).split()
        word_freq = Counter(all_words).most_common(15)
        words, counts = zip(*word_freq)
        plt.barh(range(len(words)), counts, color='lightgreen', edgecolor='black')
        plt.yticks(range(len(words)), words)
        plt.xlabel('Frequency')
        plt.title('Top 15 Most Frequent Words', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # 5. Sentiment pie chart
        ax5 = plt.subplot(2, 3, 5)
        sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=colors)
        plt.title('Sentiment Proportions', fontsize=14, fontweight='bold')
        plt.ylabel('')
        
        # 6. Character length by sentiment
        ax6 = plt.subplot(2, 3, 6)
        self.df.boxplot(column='text_length', by='sentiment', ax=ax6, patch_artist=True)
        plt.title('Character Length by Sentiment', fontsize=14, fontweight='bold')
        plt.suptitle('')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Characters')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualizations saved: {save_path}")
        plt.show()
    
    def create_wordclouds(self, save_path: str = 'twitter_wordclouds.png') -> None:
        """
        Create word clouds for each sentiment.
        
        Args:
            save_path: Path to save the word clouds
        """
        if self.df is None:
            raise ValueError("Data not processed. Complete preprocessing first.")
        
        print("\n☁️  Creating word clouds...")
        
        fig = plt.figure(figsize=(18, 6))
        
        sentiments = self.df['sentiment'].unique()
        for idx, sent in enumerate(sentiments[:3], 1):
            ax = plt.subplot(1, 3, idx)
            text_data = ' '.join(self.df[self.df['sentiment'] == sent]['final_text'])
            
            if len(text_data) > 0:
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap='Set2',
                    max_words=100
                ).generate(text_data)
                
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'WordCloud - {sent.upper()}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Word clouds saved: {save_path}")
        plt.show()
    
    def prepare_features(self) -> Tuple[Any, pd.Series]:
        """
        Prepare features for machine learning.
        
        Returns:
            Tuple of (features, target)
        """
        if self.df is None:
            raise ValueError("Data not processed. Complete preprocessing first.")
        
        print("\n🔢 Preparing features for machine learning...")
        
        # TF-IDF Vectorization
        max_features = min(5000, len(self.df) * 10)
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=1,
            max_df=0.9
        )
        
        print("🔧 Vectorizing text with TF-IDF...")
        X = self.vectorizer.fit_transform(self.df['final_text'])
        y = self.df['sentiment']
        
        print(f"✅ Feature matrix: {X.shape}")
        print(f"✅ Target: {y.shape}")
        print(f"✅ Features: {X.shape[1]}")
        
        # Show feature examples
        feature_names = self.vectorizer.get_feature_names_out()
        print(f"\n📝 Sample features: {list(feature_names[:10])}")
        
        return X, y
    
    def train_models(self, X: Any, y: pd.Series) -> Dict:
        """
        Train multiple machine learning models.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary of trained models and results
        """
        print("\n🤖 Training machine learning models...")
        
        # Split data
        min_samples = y.value_counts().min()
        test_size = 0.2
        
        if len(y) < 30 or min_samples < 3:
            print(f"⚠️  Small dataset ({len(y)} samples). Adjusting parameters...")
            test_size = 0.3 if len(y) >= 10 else 0.2
            stratify_param = None
        else:
            stratify_param = y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify_param
        )
        
        print(f"✅ Train set: {X_train.shape}")
        print(f"✅ Test set: {X_test.shape}")
        
        # Define models
        models_to_train = {
            'Naive Bayes': MultinomialNB(alpha=1.0),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
            'Linear SVM': LinearSVC(random_state=42, max_iter=1000)
        }
        
        # Train models
        self.results = {
            'X_test': X_test,
            'y_test': y_test,
            'X_train': X_train,
            'y_train': y_train
        }
        
        for model_name, model in models_to_train.items():
            print(f"\n📊 Training {model_name}...")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            self.results[model_name] = {
                'model': model,
                'predictions': y_pred,
                'accuracy': accuracy,
                'f1': f1
            }
            
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   F1-Score: {f1:.4f}")
        
        # Find best model
        self.best_model_name = max(self.results.keys(), 
                                  key=lambda x: self.results[x]['accuracy'] if x not in ['X_test', 'y_test', 'X_train', 'y_train'] else 0)
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\n🏆 Best model: {self.best_model_name}")
        print(f"   Accuracy: {self.results[self.best_model_name]['accuracy']:.4f}")
        
        return self.results
    
    def evaluate_models(self) -> None:
        """
        Perform comprehensive model evaluation.
        """
        if not self.results:
            raise ValueError("Models not trained. Call train_models() first.")
        
        print("\n📊 Model Evaluation Results:")
        
        for model_name, model_info in self.results.items():
            if model_name in ['X_test', 'y_test', 'X_train', 'y_train']:
                continue
                
            print(f"\n{model_name}:")
            print(f"  Accuracy: {model_info['accuracy']:.4f}")
            print(f"  F1-Score: {model_info['f1']:.4f}")
            
            print(f"\n  Classification Report:")
            print(classification_report(
                self.results['y_test'], 
                model_info['predictions']
            ))
    
    def create_model_visualizations(self, save_path: str = 'twitter_model_results.png') -> None:
        """
        Create model comparison visualizations.
        
        Args:
            save_path: Path to save the visualization
        """
        if not self.results:
            raise ValueError("Models not trained. Call train_models() first.")
        
        print("\n📊 Creating model comparison visualizations...")
        
        fig = plt.figure(figsize=(18, 12))
        
        # Get model names (excluding data keys)
        model_names = [name for name in self.results.keys() 
                      if name not in ['X_test', 'y_test', 'X_train', 'y_train']]
        
        # 1-3. Confusion matrices
        for idx, model_name in enumerate(model_names[:3], 1):
            ax = plt.subplot(2, 3, idx)
            cm = confusion_matrix(self.results['y_test'], self.results[model_name]['predictions'])
            
            colors = ['Blues', 'Greens', 'Purples']
            sns.heatmap(cm, annot=True, fmt='d', cmap=colors[idx-1],
                       xticklabels=np.unique(self.results['y_test']), 
                       yticklabels=np.unique(self.results['y_test']))
            plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
            plt.ylabel('True Class')
            plt.xlabel('Predicted Class')
        
        # 4. Accuracy comparison
        ax4 = plt.subplot(2, 3, 4)
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        colors_bar = ['#3498db', '#2ecc71', '#9b59b6']
        
        bars = plt.bar(model_names, accuracies, color=colors_bar, edgecolor='black', linewidth=1.5)
        plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, acc + 0.02, 
                    f'{acc:.3f}', ha='center', fontweight='bold', fontsize=11)
        
        # 5. F1-Score comparison
        ax5 = plt.subplot(2, 3, 5)
        f1_scores = [self.results[name]['f1'] for name in model_names]
        
        bars = plt.bar(model_names, f1_scores, color=colors_bar, edgecolor='black', linewidth=1.5)
        plt.title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('F1-Score (weighted)')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        for bar, f1 in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, f1 + 0.02,
                    f'{f1:.3f}', ha='center', fontweight='bold', fontsize=11)
        
        # 6. Feature importance (Logistic Regression)
        ax6 = plt.subplot(2, 3, 6)
        if 'Logistic Regression' in model_names:
            lr_model = self.results['Logistic Regression']['model']
            if hasattr(lr_model, 'coef_'):
                coef_index = 0 if len(lr_model.coef_) > 0 else 0
                feature_names = self.vectorizer.get_feature_names_out()
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.abs(lr_model.coef_[coef_index])
                }).sort_values('importance', ascending=False).head(15)
                
                plt.barh(range(len(feature_importance)), feature_importance['importance'], 
                        color='coral')
                plt.yticks(range(len(feature_importance)), feature_importance['feature'])
                plt.xlabel('Importance (absolute value)')
                plt.title('Top 15 Important Features (LR)', fontsize=14, fontweight='bold')
                plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Model visualizations saved: {save_path}")
        plt.show()
    
    def predict_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment for a given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (predicted_sentiment, confidence)
        """
        if self.best_model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        # Preprocess text
        cleaned_text = self.clean_text(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        cleaned_text = ' '.join([word for word in cleaned_text.split() 
                                if word not in stop_words and len(word) > 2])
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        cleaned_text = ' '.join([lemmatizer.lemmatize(word) for word in cleaned_text.split()])
        
        # Vectorize and predict
        text_vectorized = self.vectorizer.transform([cleaned_text])
        prediction = self.best_model.predict(text_vectorized)[0]
        
        # Get confidence
        if hasattr(self.best_model, 'predict_proba'):
            confidence = self.best_model.predict_proba(text_vectorized)[0].max()
        else:
            confidence = 1.0
        
        return prediction, confidence
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete Twitter sentiment analysis pipeline.
        
        Returns:
            Dictionary containing analysis results
        """
        print("🐦 Twitter Sentiment Analysis Pipeline")
        print("=" * 70)
        
        try:
            # Load and explore data
            self.load_data()
            
            # Preprocess text
            self.preprocess_text()
            
            # Create visualizations
            self.create_visualizations()
            self.create_wordclouds()
            
            # Prepare features
            X, y = self.prepare_features()
            
            # Train models
            self.train_models(X, y)
            
            # Evaluate models
            self.evaluate_models()
            
            # Create model visualizations
            self.create_model_visualizations()
            
            print("\n🎉 Complete analysis finished successfully!")
            
            return {
                'dataset_size': len(self.df),
                'best_model': self.best_model_name,
                'best_accuracy': self.results[self.best_model_name]['accuracy'],
                'best_f1': self.results[self.best_model_name]['f1'],
                'results': self.results
            }
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            raise


def main():
    """
    Main function to run the Twitter sentiment analysis.
    """
    analyzer = TwitterSentimentAnalyzer('tweets.csv')
    results = analyzer.run_complete_analysis()
    
    # Print summary
    print(f"\n📊 Analysis Summary:")
    print(f"Dataset size: {results['dataset_size']} tweets")
    print(f"Best model: {results['best_model']}")
    print(f"Accuracy: {results['best_accuracy']:.4f}")
    print(f"F1-Score: {results['best_f1']:.4f}")
    
    # Example predictions
    print(f"\n🧪 Example predictions:")
    test_tweets = [
        "This airline is amazing! Best flight ever 😊",
        "Terrible service, my flight was delayed for 5 hours",
        "The flight was okay, nothing special"
    ]
    
    for tweet in test_tweets:
        prediction, confidence = analyzer.predict_sentiment(tweet)
        print(f"Tweet: '{tweet[:50]}...'")
        print(f"→ Sentiment: {prediction}")
        print(f"→ Confidence: {confidence:.2f}")
        print()


if __name__ == "__main__":
    main()
