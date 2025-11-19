"""
House Price Prediction System

A comprehensive machine learning system for predicting house prices using
multiple regression algorithms with extensive analysis and visualization.

Author: Chaghoub Aymen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import pickle
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                             r2_score, mean_absolute_percentage_error)


class HousePricePredictor:
    """
    A comprehensive house price prediction system.
    
    This class handles data loading, preprocessing, feature engineering,
    model training, evaluation, and prediction for house price analysis.
    """
    
    def __init__(self, data_path: str = 'USA_Housing.csv'):
        """
        Initialize the HousePricePredictor.
        
        Args:
            data_path: Path to the housing dataset CSV file
        """
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models: Dict = {}
        self.results: Dict = {}
        self.best_model = None
        self.best_model_name = None
        
        # Configure plotting
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and explore the housing dataset.
        
        Returns:
            Loaded DataFrame
        """
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        except FileNotFoundError:
            print(f"⚠️  File {self.data_path} not found. Creating sample data...")
            self._create_sample_data()
        
        print(f"\n📋 Dataset overview:")
        print(self.df.head(10))
        print(f"\n🔍 Data types:")
        print(self.df.info())
        print(f"\n📊 Basic statistics:")
        print(self.df.describe())
        
        # Check missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            missing_pct = (missing / len(self.df)) * 100
            missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
            print(f"\n❌ Missing values:")
            print(missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False))
        else:
            print(f"\n✅ No missing values found")
        
        # Check duplicates
        duplicates = self.df.duplicated().sum()
        print(f"\n🔄 Duplicates: {duplicates}")
        if duplicates > 0:
            self.df = self.df.drop_duplicates()
            print(f"✅ Duplicates removed")
        
        return self.df
    
    def _create_sample_data(self) -> None:
        """
        Create realistic sample housing data for demonstration.
        """
        np.random.seed(42)
        n_samples = 5000
        
        # Generate realistic housing features
        avg_area_income = np.random.normal(68000, 15000, n_samples)
        avg_area_house_age = np.random.uniform(2, 30, n_samples)
        avg_area_rooms = np.random.normal(6.5, 1.5, n_samples)
        avg_area_bedrooms = np.random.normal(3.5, 0.8, n_samples)
        area_population = np.random.normal(36000, 8000, n_samples)
        
        # Generate realistic prices with some correlation
        price = (
            avg_area_income * 0.8 +
            avg_area_house_age * -5000 +
            avg_area_rooms * 150000 +
            avg_area_bedrooms * 50000 +
            area_population * 0.5 +
            np.random.normal(0, 50000, n_samples)
        )
        
        # Ensure positive prices
        price = np.maximum(price, 100000)
        
        self.df = pd.DataFrame({
            'Avg. Area Income': avg_area_income,
            'Avg. Area House Age': avg_area_house_age,
            'Avg. Area Number of Rooms': avg_area_rooms,
            'Avg. Area Number of Bedrooms': avg_area_bedrooms,
            'Area Population': area_population,
            'Price': price,
            'Address': [f'{np.random.randint(100, 9999)} Main St, City {i%50}' 
                       for i in range(n_samples)]
        })
        
        print(f"✅ Sample data created: {self.df.shape[0]} records")
    
    def explore_data(self) -> None:
        """
        Perform comprehensive exploratory data analysis.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n📊 Exploratory Data Analysis")
        print("=" * 50)
        
        # Identify numeric columns and target
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\n✅ Numeric columns: {len(numeric_cols)}")
        print(f"   {numeric_cols}")
        
        # Identify target column
        target_col = None
        for col in self.df.columns:
            if 'price' in col.lower() or 'saleprice' in col.lower():
                target_col = col
                break
        
        if target_col is None:
            target_col = numeric_cols[-1]
        
        print(f"\n🎯 Target variable: '{target_col}'")
        
        # Price statistics
        print(f"\n💰 Price statistics:")
        print(f"   Mean:    ${self.df[target_col].mean():,.2f}")
        print(f"   Median:  ${self.df[target_col].median():,.2f}")
        print(f"   Min:     ${self.df[target_col].min():,.2f}")
        print(f"   Max:     ${self.df[target_col].max():,.2f}")
        print(f"   Std:     ${self.df[target_col].std():,.2f}")
        
        # Create visualizations
        self._create_eda_visualizations(target_col, numeric_cols)
        
        # Correlation analysis
        correlation_matrix = self.df[numeric_cols].corr()
        price_correlations = correlation_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
        
        print(f"\n📊 Top correlations with price:")
        for feature, corr in price_correlations.head(5).items():
            print(f"   {feature:40s}: {corr:+.4f}")
    
    def _create_eda_visualizations(self, target_col: str, numeric_cols: List[str]) -> None:
        """
        Create exploratory data analysis visualizations.
        
        Args:
            target_col: Name of the target column
            numeric_cols: List of numeric column names
        """
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Price distribution
        ax1 = plt.subplot(2, 3, 1)
        plt.hist(self.df[target_col], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(self.df[target_col].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: ${self.df[target_col].mean():,.0f}')
        plt.axvline(self.df[target_col].median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: ${self.df[target_col].median():,.0f}')
        plt.title('Price Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Price ($)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.ticklabel_format(style='plain', axis='x')
        
        # 2. Price boxplot
        ax2 = plt.subplot(2, 3, 2)
        plt.boxplot(self.df[target_col], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightcoral', alpha=0.7))
        plt.title('Price Boxplot', fontsize=14, fontweight='bold')
        plt.ylabel('Price ($)')
        plt.ticklabel_format(style='plain', axis='y')
        
        # 3. Correlation heatmap
        ax3 = plt.subplot(2, 3, 3)
        correlation_matrix = self.df[numeric_cols].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        # 4-6. Scatter plots for top correlated features
        price_correlations = correlation_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
        top_features = price_correlations.head(3).index.tolist()
        
        for idx, feature in enumerate(top_features, 4):
            ax = plt.subplot(2, 3, idx)
            plt.scatter(self.df[feature], self.df[target_col], alpha=0.5, s=20, color='mediumpurple')
            
            # Add trend line
            z = np.polyfit(self.df[feature], self.df[target_col], 1)
            p = np.poly1d(z)
            plt.plot(self.df[feature], p(self.df[feature]), "r--", linewidth=2, alpha=0.8)
            
            correlation = self.df[feature].corr(self.df[target_col])
            plt.title(f'{feature} vs Price\n(r = {correlation:.3f})', 
                     fontsize=12, fontweight='bold')
            plt.xlabel(feature)
            plt.ylabel('Price ($)')
            plt.ticklabel_format(style='plain', axis='y')
        
        plt.tight_layout()
        plt.savefig('house_eda.png', dpi=300, bbox_inches='tight')
        print("\n✅ EDA visualizations saved: house_eda.png")
        plt.show()
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for machine learning modeling.
        
        Returns:
            Tuple of (features, target)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n🔧 Preparing data for modeling...")
        
        self.df_clean = self.df.copy()
        
        # Remove non-numeric columns
        non_numeric_cols = self.df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            print(f"🗑️  Removing non-numeric columns: {non_numeric_cols}")
            self.df_clean = self.df_clean.drop(columns=non_numeric_cols)
        
        # Handle missing values
        if self.df_clean.isnull().sum().sum() > 0:
            print("🔧 Handling missing values...")
            self.df_clean = self.df_clean.fillna(self.df_clean.median())
            print("✅ Missing values filled with median")
        
        # Detect outliers (informational)
        print("\n🔍 Outlier detection (IQR method)...")
        outliers_count = 0
        for col in self.df_clean.columns:
            Q1 = self.df_clean[col].quantile(0.25)
            Q3 = self.df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            outliers = ((self.df_clean[col] < lower_bound) | (self.df_clean[col] > upper_bound)).sum()
            outliers_count += outliers
        
        print(f"   Total outliers detected: {outliers_count}")
        print(f"   (Outliers kept for this analysis)")
        
        # Identify target column
        target_col = None
        for col in self.df_clean.columns:
            if 'price' in col.lower() or 'saleprice' in col.lower():
                target_col = col
                break
        
        if target_col is None:
            target_col = self.df_clean.columns[-1]
        
        # Separate features and target
        self.X = self.df_clean.drop(target_col, axis=1)
        self.y = self.df_clean[target_col]
        
        print(f"\n✅ Features (X): {self.X.shape}")
        print(f"✅ Target (y): {self.y.shape}")
        print(f"\n📋 Features used:")
        for i, col in enumerate(self.X.columns, 1):
            print(f"   {i}. {col}")
        
        return self.X, self.y
    
    def train_models(self, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train multiple machine learning models.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary of trained models and results
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        print("\n🤖 Training machine learning models...")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        print(f"✅ Train set: {self.X_train.shape[0]} samples ({int((1-test_size)*100)}%)")
        print(f"✅ Test set: {self.X_test.shape[0]} samples ({int(test_size*100)}%)")
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Data standardized")
        
        # Define models to train
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, max_depth=20, min_samples_split=5,
                random_state=random_state, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                random_state=random_state
            )
        }
        
        # Train models
        self.results = {}
        
        for model_name, model in models_to_train.items():
            print(f"\n📊 Training {model_name}...")
            
            # Use scaled data for Ridge, original for others
            if 'Ridge' in model_name:
                model.fit(X_train_scaled, self.y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)
            mape = mean_absolute_percentage_error(self.y_test, y_pred) * 100
            
            self.results[model_name] = {
                'model': model,
                'predictions': y_pred,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            }
            
            print(f"   MAE:  ${mae:,.2f}")
            print(f"   RMSE: ${rmse:,.2f}")
            print(f"   R²:   {r2:.4f}")
            print(f"   MAPE: {mape:.2f}%")
        
        # Find best model
        self.best_model_name = max(self.results, key=lambda x: self.results[x]['R2'])
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\n🏆 Best model: {self.best_model_name}")
        print(f"   R² Score: {self.results[self.best_model_name]['R2']:.4f}")
        
        return self.results
    
    def evaluate_models(self) -> None:
        """
        Perform comprehensive model evaluation and comparison.
        """
        if not self.results:
            raise ValueError("Models not trained. Call train_models() first.")
        
        print("\n📊 Model Evaluation and Comparison")
        print("=" * 50)
        
        # Cross-validation
        print("\n🔄 Cross-validation results (5-fold):")
        for model_name, model_info in self.results.items():
            model = model_info['model']
            
            # Use appropriate data for CV
            if 'Ridge' in model_name:
                cv_scores = cross_val_score(model, self.scaler.transform(self.X_train), 
                                          self.y_train, cv=5, scoring='r2', n_jobs=-1)
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=5, scoring='r2', n_jobs=-1)
            
            print(f"   {model_name:20s}: R² = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Detailed error analysis
        best_predictions = self.results[self.best_model_name]['predictions']
        errors = self.y_test - best_predictions
        percentage_errors = (errors / self.y_test) * 100
        
        print(f"\n🔍 Detailed error analysis ({self.best_model_name}):")
        print(f"   Mean absolute error:     ${abs(errors).mean():,.2f}")
        print(f"   Median error:           ${errors.median():,.2f}")
        print(f"   Error std:              ${errors.std():,.2f}")
        print(f"   Max overestimation:     ${errors.max():,.2f}")
        print(f"   Max underestimation:    ${errors.min():,.2f}")
        print(f"   Mean percentage error:  {abs(percentage_errors).mean():.2f}%")
        
        # Accuracy within ranges
        within_10 = (abs(percentage_errors) <= 10).sum()
        within_20 = (abs(percentage_errors) <= 20).sum()
        
        print(f"   Predictions within ±10%: {within_10} ({within_10/len(percentage_errors)*100:.1f}%)")
        print(f"   Predictions within ±20%: {within_20} ({within_20/len(percentage_errors)*100:.1f}%)")
    
    def create_visualizations(self, save_path: str = 'house_model_results.png') -> None:
        """
        Create comprehensive model comparison visualizations.
        
        Args:
            save_path: Path to save the visualization
        """
        if not self.results:
            raise ValueError("Models not trained. Call train_models() first.")
        
        print("\n📊 Creating model comparison visualizations...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. R² comparison
        ax1 = plt.subplot(2, 3, 1)
        model_names = list(self.results.keys())
        r2_scores = [self.results[m]['R2'] for m in model_names]
        colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        bars = plt.bar(model_names, r2_scores, color=colors_bar, edgecolor='black', linewidth=1.5)
        plt.title('R² Score Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('R² Score')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        for bar, r2 in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, r2 + 0.02,
                    f'{r2:.4f}', ha='center', fontweight='bold', fontsize=10)
        
        # 2. RMSE comparison
        ax2 = plt.subplot(2, 3, 2)
        rmse_scores = [self.results[m]['RMSE'] for m in model_names]
        bars = plt.bar(model_names, rmse_scores, color=colors_bar, edgecolor='black', linewidth=1.5)
        plt.title('RMSE Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('RMSE ($)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        for bar, rmse in zip(bars, rmse_scores):
            plt.text(bar.get_x() + bar.get_width()/2, rmse + max(rmse_scores)*0.01,
                    f'${rmse:,.0f}', ha='center', fontweight='bold', fontsize=9, rotation=90)
        
        # 3. MAPE comparison
        ax3 = plt.subplot(2, 3, 3)
        mape_scores = [self.results[m]['MAPE'] for m in model_names]
        bars = plt.bar(model_names, mape_scores, color=colors_bar, edgecolor='black', linewidth=1.5)
        plt.title('MAPE Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('MAPE (%)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        for bar, mape in zip(bars, mape_scores):
            plt.text(bar.get_x() + bar.get_width()/2, mape + max(mape_scores)*0.01,
                    f'{mape:.2f}%', ha='center', fontweight='bold', fontsize=9)
        
        # 4. Predictions vs actual (best model)
        ax4 = plt.subplot(2, 3, 4)
        best_predictions = self.results[self.best_model_name]['predictions']
        plt.scatter(self.y_test, best_predictions, alpha=0.5, s=30, color='mediumpurple')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        plt.title(f'Predictions vs Actual - {self.best_model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.legend()
        plt.ticklabel_format(style='plain')
        plt.grid(alpha=0.3)
        
        # 5. Residuals analysis
        ax5 = plt.subplot(2, 3, 5)
        residuals = self.y_test - best_predictions
        plt.scatter(best_predictions, residuals, alpha=0.5, s=30, color='coral')
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.title(f'Residuals Analysis - {self.best_model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Price ($)')
        plt.ylabel('Residuals ($)')
        plt.ticklabel_format(style='plain')
        plt.grid(alpha=0.3)
        
        # 6. Feature importance
        ax6 = plt.subplot(2, 3, 6)
        if hasattr(self.best_model, 'feature_importances_'):
            # Tree-based model
            importances = self.best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False).head(10)
            
            plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'], 
                    color='lightgreen')
            plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
            plt.xlabel('Importance')
            plt.title(f'Top Features - {self.best_model_name}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
        else:
            # Linear model coefficients
            if hasattr(self.best_model, 'coef_'):
                coef_df = pd.DataFrame({
                    'feature': self.X.columns,
                    'coefficient': np.abs(self.best_model.coef_)
                }).sort_values('coefficient', ascending=False).head(10)
                
                plt.barh(range(len(coef_df)), coef_df['coefficient'], color='lightblue')
                plt.yticks(range(len(coef_df)), coef_df['feature'])
                plt.xlabel('Coefficient (absolute value)')
                plt.title(f'Top Coefficients - {self.best_model_name}', fontsize=14, fontweight='bold')
                plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Model visualizations saved: {save_path}")
        plt.show()
    
    def predict_price(self, features_dict: Dict[str, float]) -> float:
        """
        Predict house price for given features.
        
        Args:
            features_dict: Dictionary with feature values
            
        Returns:
            Predicted price
        """
        if self.best_model is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        # Create DataFrame with features in correct order
        features_df = pd.DataFrame([features_dict])
        
        # Ensure all required columns are present
        for col in self.X.columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Reorder columns
        features_df = features_df[self.X.columns]
        
        # Make prediction
        if 'Ridge' in self.best_model_name:
            features_scaled = self.scaler.transform(features_df)
            prediction = self.best_model.predict(features_scaled)[0]
        else:
            prediction = self.best_model.predict(features_df)[0]
        
        return prediction
    
    def save_model(self, model_path: str = 'best_house_price_model.pkl') -> None:
        """
        Save the best trained model and metadata.
        
        Args:
            model_path: Path to save the model
        """
        if self.best_model is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save metadata
        metadata = {
            'feature_names': self.X.columns.tolist(),
            'model_name': self.best_model_name,
            'r2_score': self.results[self.best_model_name]['R2'],
            'mae': self.results[self.best_model_name]['MAE'],
            'rmse': self.results[self.best_model_name]['RMSE'],
            'scaler': self.scaler
        }
        
        metadata_path = 'model_metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Metadata saved: {metadata_path}")
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            raise ValueError("Models not trained. Call train_models() first.")
        
        report = f"""
================================================================================
        HOUSE PRICE PREDICTION ANALYSIS REPORT
================================================================================

📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. DATASET OVERVIEW
------------------
• Total samples: {len(self.df)}
• Features used: {len(self.X.columns)}
• Price range: ${self.y.min():,.2f} - ${self.y.max():,.2f}
• Average price: ${self.y.mean():,.2f}

2. MODEL PERFORMANCE COMPARISON
-------------------------------
"""
        
        for model_name, metrics in self.results.items():
            report += f"""
{model_name:25s}:
   • R² Score: {metrics['R2']:.4f}
   • RMSE:     ${metrics['RMSE']:,.2f}
   • MAE:      ${metrics['MAE']:,.2f}
   • MAPE:     {metrics['MAPE']:.2f}%
"""
        
        best_metrics = self.results[self.best_model_name]
        report += f"""
3. BEST MODEL: {self.best_model_name}
{'='*len(self.best_model_name) + '='*20}
• R² Score: {best_metrics['R2']:.4f}
• RMSE:     ${best_metrics['RMSE']:,.2f}
• MAE:      ${best_metrics['MAE']:,.2f}
• MAPE:     {best_metrics['MAPE']:.2f}%

4. INTERPRETATION
-----------------
The model explains {best_metrics['R2']*100:.2f}% of the variance in house prices.
Average prediction error: ${best_metrics['MAE']:,.0f}
Relative error: {best_metrics['MAPE']:.2f}%

5. FEATURES USED
----------------
"""
        
        for i, feature in enumerate(self.X.columns, 1):
            report += f"   {i}. {feature}\n"
        
        report += """
================================================================================
                    END OF REPORT
================================================================================
"""
        
        return report
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete house price prediction analysis pipeline.
        
        Returns:
            Dictionary containing analysis results
        """
        print("🏠 House Price Prediction Analysis Pipeline")
        print("=" * 70)
        
        try:
            # Load and explore data
            self.load_data()
            
            # Exploratory data analysis
            self.explore_data()
            
            # Prepare data for modeling
            self.prepare_data()
            
            # Train models
            self.train_models()
            
            # Evaluate models
            self.evaluate_models()
            
            # Create visualizations
            self.create_visualizations()
            
            # Save model
            self.save_model()
            
            # Generate report
            report = self.generate_report()
            with open('rapport_final.txt', 'w', encoding='utf-8') as f:
                f.write(report)
            print("\n✅ Final report saved: rapport_final.txt")
            
            print("\n🎉 Complete analysis finished successfully!")
            
            return {
                'dataset_size': len(self.df),
                'best_model': self.best_model_name,
                'best_r2': self.results[self.best_model_name]['R2'],
                'best_mae': self.results[self.best_model_name]['MAE'],
                'results': self.results
            }
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            raise


def main():
    """
    Main function to run the house price prediction analysis.
    """
    predictor = HousePricePredictor('USA_Housing.csv')
    results = predictor.run_complete_analysis()
    
    # Print summary
    print(f"\n📊 Analysis Summary:")
    print(f"Dataset size: {results['dataset_size']} houses")
    print(f"Best model: {results['best_model']}")
    print(f"R² Score: {results['best_r2']:.4f}")
    print(f"MAE: ${results['best_mae']:,.2f}")
    
    # Example prediction
    print(f"\n🏠 Example prediction:")
    example_house = {
        'Avg. Area Income': 75000,
        'Avg. Area House Age': 10,
        'Avg. Area Number of Rooms': 7,
        'Avg. Area Number of Bedrooms': 4,
        'Area Population': 35000
    }
    
    predicted_price = predictor.predict_price(example_house)
    print(f"Features: {example_house}")
    print(f"Predicted price: ${predicted_price:,.2f}")


if __name__ == "__main__":
    main()
