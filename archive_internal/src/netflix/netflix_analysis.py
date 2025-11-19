"""
Netflix Content Analysis and Popularity Prediction

This module analyzes Netflix content data and predicts content popularity
using various machine learning algorithms.

Author: Chaghoub Aymen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path

warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc


class NetflixAnalyzer:
    """
    A comprehensive Netflix content analyzer and popularity predictor.
    
    This class handles data loading, preprocessing, feature engineering,
    model training, and evaluation for Netflix content analysis.
    """
    
    def __init__(self, data_path: str = 'netflix_titles.csv'):
        """
        Initialize the NetflixAnalyzer.
        
        Args:
            data_path: Path to the Netflix dataset CSV file
        """
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        self.df_model: Optional[pd.DataFrame] = None
        self.models: Dict = {}
        self.results: Dict = {}
        self.scaler = StandardScaler()
        
        # Configure plotting
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and explore the Netflix dataset.
        
        Returns:
            Loaded DataFrame
        """
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            print(f"\n📋 First few rows:")
            print(self.df.head())
            print(f"\n🔍 Data types:")
            print(self.df.info())
            print(f"\n📈 Basic statistics:")
            print(self.df.describe())
            
            # Check missing values
            missing = self.df.isnull().sum()
            missing_pct = (missing / len(self.df)) * 100
            missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
            print(f"\n❌ Missing values:")
            print(missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False))
            
            # Check duplicates
            duplicates = self.df.duplicated().sum()
            print(f"\n🔄 Duplicates: {duplicates}")
            
            return self.df
        except FileNotFoundError:
            print(f"❌ File {self.data_path} not found!")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the Netflix dataset.
        
        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n🧹 Cleaning and preprocessing data...")
        self.df_clean = self.df.copy()
        
        # Remove duplicates
        initial_size = len(self.df_clean)
        self.df_clean = self.df_clean.drop_duplicates()
        print(f"✅ Removed {initial_size - len(self.df_clean)} duplicates")
        
        # Handle missing values
        self.df_clean['director'] = self.df_clean['director'].fillna('Unknown')
        self.df_clean['cast'] = self.df_clean['cast'].fillna('Unknown')
        self.df_clean['country'] = self.df_clean['country'].fillna('Unknown')
        self.df_clean['rating'] = self.df_clean['rating'].fillna('Not Rated')
        
        # Remove rows with missing date_added
        self.df_clean = self.df_clean.dropna(subset=['date_added'])
        print(f"✅ Handled missing values")
        print(f"✅ Final dataset size: {self.df_clean.shape}")
        
        return self.df_clean
    
    def extract_duration(self, duration: str, content_type: str) -> Optional[int]:
        """
        Extract numeric duration from duration string.
        
        Args:
            duration: Duration string (e.g., "90 min", "2 Seasons")
            content_type: Type of content ("Movie" or "TV Show")
            
        Returns:
            Numeric duration value
        """
        if pd.isna(duration):
            return np.nan
        try:
            return int(duration.split()[0])
        except (ValueError, IndexError):
            return np.nan
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Create new features for machine learning.
        
        Returns:
            DataFrame with engineered features
        """
        if self.df_clean is None:
            raise ValueError("Data not cleaned. Call clean_data() first.")
        
        print("\n🔧 Engineering features...")
        
        # Convert dates
        self.df_clean['date_added'] = pd.to_datetime(
            self.df_clean['date_added'].str.strip(), 
            format='%B %d, %Y', 
            errors='coerce'
        )
        self.df_clean['year_added'] = self.df_clean['date_added'].dt.year
        self.df_clean['month_added'] = self.df_clean['date_added'].dt.month
        
        # Extract duration
        self.df_clean['duration_value'] = self.df_clean.apply(
            lambda x: self.extract_duration(x['duration'], x['type']), axis=1
        )
        
        # Extract primary country
        self.df_clean['primary_country'] = self.df_clean['country'].apply(
            lambda x: x.split(',')[0].strip() if pd.notna(x) else 'Unknown'
        )
        
        # Count cast members
        self.df_clean['cast_count'] = self.df_clean['cast'].apply(
            lambda x: 0 if x == 'Unknown' else len(x.split(','))
        )
        
        # Count genres
        self.df_clean['genre_count'] = self.df_clean['listed_in'].apply(
            lambda x: len(x.split(',')) if pd.notna(x) else 0
        )
        
        # Extract primary genre
        self.df_clean['primary_genre'] = self.df_clean['listed_in'].apply(
            lambda x: x.split(',')[0].strip() if pd.notna(x) else 'Unknown'
        )
        
        print("✅ Features created: year_added, month_added, duration_value, "
              "primary_country, cast_count, genre_count, primary_genre")
        
        return self.df_clean
    
    def create_target_variable(self) -> pd.DataFrame:
        """
        Create the target variable for popularity prediction.
        
        Returns:
            DataFrame with target variable
        """
        if self.df_clean is None:
            raise ValueError("Features not engineered. Call engineer_features() first.")
        
        print("\n🎯 Creating target variable (is_popular)...")
        
        # Define popular genres
        popular_genres = ['International Movies', 'Dramas', 'Comedies', 
                         'Action & Adventure', 'Documentaries']
        
        # Create popularity criteria
        self.df_clean['is_popular'] = (
            ((self.df_clean['release_year'] >= 2015) & (self.df_clean['type'] == 'Movie')) |
            (self.df_clean['primary_genre'].isin(popular_genres)) |
            ((self.df_clean['duration_value'] >= 80) & 
             (self.df_clean['duration_value'] <= 120) & 
             (self.df_clean['type'] == 'Movie'))
        ).astype(int)
        
        print(f"✅ Popular content: {self.df_clean['is_popular'].sum()} "
              f"({self.df_clean['is_popular'].mean() * 100:.1f}%)")
        
        return self.df_clean
    
    def prepare_modeling_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for machine learning modeling.
        
        Returns:
            Tuple of (features, target)
        """
        if self.df_clean is None:
            raise ValueError("Target variable not created. Call create_target_variable() first.")
        
        print("\n📊 Preparing data for modeling...")
        
        # Select features
        features_to_encode = ['type', 'rating', 'primary_country', 'primary_genre']
        numerical_features = ['release_year', 'duration_value', 'cast_count', 
                             'genre_count', 'year_added', 'month_added']
        
        # Create modeling dataset
        self.df_model = self.df_clean[features_to_encode + numerical_features + ['is_popular']].copy()
        self.df_model = self.df_model.dropna()
        
        print(f"✅ Modeling dataset: {self.df_model.shape}")
        
        # Encode categorical variables
        le_dict = {}
        for col in features_to_encode:
            le = LabelEncoder()
            self.df_model[f'{col}_encoded'] = le.fit_transform(self.df_model[col])
            le_dict[col] = le
            print(f"✅ Encoded {col}: {self.df_model[f'{col}_encoded'].nunique()} classes")
        
        # Select final features
        feature_columns = [f'{col}_encoded' for col in features_to_encode] + numerical_features
        X = self.df_model[feature_columns]
        y = self.df_model['is_popular']
        
        print(f"✅ Final features: {X.shape}")
        print(f"✅ Target: {y.shape}")
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train multiple machine learning models.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary of trained models
        """
        print("\n🤖 Training machine learning models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Train set: {X_train.shape}")
        print(f"✅ Test set: {X_test.shape}")
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print("\n🌲 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=15, min_samples_split=5,
            random_state=42, n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        # Train Logistic Regression
        print("📈 Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        y_pred_lr = lr_model.predict(X_test_scaled)
        
        # Store models and results
        self.models = {
            'Random Forest': rf_model,
            'Logistic Regression': lr_model
        }
        
        self.results = {
            'X_test': X_test,
            'y_test': y_test,
            'X_test_scaled': X_test_scaled,
            'Random Forest': {
                'predictions': y_pred_rf,
                'accuracy': accuracy_score(y_test, y_pred_rf),
                'f1': f1_score(y_test, y_pred_rf)
            },
            'Logistic Regression': {
                'predictions': y_pred_lr,
                'accuracy': accuracy_score(y_test, y_pred_lr),
                'f1': f1_score(y_test, y_pred_lr)
            }
        }
        
        print("✅ Model training completed!")
        return self.models
    
    def evaluate_models(self) -> None:
        """
        Evaluate and compare trained models.
        """
        if not self.models:
            raise ValueError("Models not trained. Call train_models() first.")
        
        print("\n📊 Model Evaluation Results:")
        
        for model_name, model in self.models.items():
            results = self.results[model_name]
            print(f"\n{model_name}:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  F1-Score: {results['f1']:.4f}")
            
            print(f"\n  Classification Report:")
            print(classification_report(
                self.results['y_test'], 
                results['predictions'],
                target_names=['Not Popular', 'Popular']
            ))
    
    def create_visualizations(self, save_path: str = 'netflix_analysis.png') -> None:
        """
        Create comprehensive visualizations of the analysis.
        
        Args:
            save_path: Path to save the visualization
        """
        if self.df_clean is None:
            raise ValueError("Data not processed. Complete the pipeline first.")
        
        print("\n📊 Creating visualizations...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Top countries
        ax1 = plt.subplot(2, 3, 1)
        top_countries = self.df_clean['primary_country'].value_counts().head(10)
        top_countries.plot(kind='barh', color='skyblue')
        plt.title('Top 10 Countries by Content', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Contents')
        plt.ylabel('Country')
        plt.gca().invert_yaxis()
        
        # 2. Content evolution by year
        ax2 = plt.subplot(2, 3, 2)
        year_counts = self.df_clean['release_year'].value_counts().sort_index()
        year_counts[year_counts.index >= 2000].plot(kind='line', linewidth=2, color='coral')
        plt.title('Content Evolution (2000+)', fontsize=14, fontweight='bold')
        plt.xlabel('Release Year')
        plt.ylabel('Number of Contents')
        plt.grid(alpha=0.3)
        
        # 3. Top genres
        ax3 = plt.subplot(2, 3, 3)
        top_genres = self.df_clean['primary_genre'].value_counts().head(10)
        top_genres.plot(kind='bar', color='lightgreen')
        plt.title('Top 10 Genres', fontsize=14, fontweight='bold')
        plt.xlabel('Genre')
        plt.ylabel('Number of Contents')
        plt.xticks(rotation=45, ha='right')
        
        # 4. Movie duration distribution
        ax4 = plt.subplot(2, 3, 4)
        movies_duration = self.df_clean[
            self.df_clean['type'] == 'Movie'
        ]['duration_value'].dropna()
        plt.hist(movies_duration, bins=30, color='mediumpurple', 
                edgecolor='black', alpha=0.7)
        plt.title('Movie Duration Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Frequency')
        
        # 5. Content type distribution
        ax5 = plt.subplot(2, 3, 5)
        type_counts = self.df_clean['type'].value_counts()
        type_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90,
                        colors=['#ff9999', '#66b3ff'])
        plt.title('Movies vs TV Shows', fontsize=14, fontweight='bold')
        plt.ylabel('')
        
        # 6. Content added by year
        ax6 = plt.subplot(2, 3, 6)
        year_added_counts = self.df_clean['year_added'].value_counts().sort_index()
        year_added_counts.plot(kind='area', color='gold', alpha=0.6, linewidth=2)
        plt.title('Content Added to Netflix by Year', fontsize=14, fontweight='bold')
        plt.xlabel('Year Added')
        plt.ylabel('Number of Contents')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualizations saved: {save_path}")
        plt.show()
    
    def run_complete_analysis(self) -> Dict:
        """
        Run the complete Netflix analysis pipeline.
        
        Returns:
            Dictionary containing analysis results
        """
        print("🎬 Netflix Content Analysis Pipeline")
        print("=" * 60)
        
        try:
            # Load and explore data
            self.load_data()
            
            # Clean and preprocess
            self.clean_data()
            
            # Engineer features
            self.engineer_features()
            
            # Create target variable
            self.create_target_variable()
            
            # Prepare modeling data
            X, y = self.prepare_modeling_data()
            
            # Train models
            self.train_models(X, y)
            
            # Evaluate models
            self.evaluate_models()
            
            # Create visualizations
            self.create_visualizations()
            
            print("\n🎉 Analysis completed successfully!")
            
            return {
                'dataset_size': len(self.df_clean),
                'models': self.models,
                'results': self.results
            }
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            raise


def main():
    """
    Main function to run the Netflix analysis.
    """
    analyzer = NetflixAnalyzer('netflix_titles.csv')
    results = analyzer.run_complete_analysis()
    
    # Print summary
    print(f"\n📊 Analysis Summary:")
    print(f"Dataset size: {results['dataset_size']} contents")
    print(f"Models trained: {len(results['models'])}")
    
    # Best model
    best_model = max(results['results'].keys(), 
                    key=lambda x: results['results'][x]['accuracy'])
    print(f"Best model: {best_model} "
          f"(Accuracy: {results['results'][best_model]['accuracy']:.4f})")


if __name__ == "__main__":
    main()
