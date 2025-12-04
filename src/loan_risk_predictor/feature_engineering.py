"""
Feature engineering module for loan risk prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


class FeatureEngineer:
    """
    Handles feature engineering for loan risk prediction
    """
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction and derived features
        """
        df = df.copy()
        
        # Income-to-Age ratio
        df['Income_Age_Ratio'] = df['Income'] / (df['Age'] + 1)
        
        # Experience-to-Age ratio
        df['Experience_Age_Ratio'] = df['Experience'] / (df['Age'] + 1)
        
        # Stability score
        df['Stability_Score'] = df['CURRENT_JOB_YRS'] + df['CURRENT_HOUSE_YRS']
        
        # DTI Ratio (simulated)
        df['DTI_Ratio'] = df['Income'] * 0.3 / (df['Income'] + 1)
        
        return df
    
    @staticmethod
    def create_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create categorical features from continuous ones
        """
        df = df.copy()
        
        # Age groups
        def categorize_age(age: int) -> str:
            if age < 25:
                return 'Young'
            elif age < 35:
                return 'Young_Adult'
            elif age < 50:
                return 'Middle_Aged'
            else:
                return 'Senior'
        
        df['Age_Group'] = df['Age'].apply(categorize_age)
        
        # Income categories
        def categorize_income(income: int) -> str:
            if income < 1000000:
                return 'Low'
            elif income < 3000000:
                return 'Medium'
            elif income < 6000000:
                return 'High'
            else:
                return 'Very_High'
        
        df['Income_Category'] = df['Income'].apply(categorize_income)
        
        # Stability categories
        def categorize_stability(score: float) -> str:
            if score < 5:
                return 'Low'
            elif score < 15:
                return 'Medium'
            else:
                return 'High'
        
        df['Stability_Category'] = df['Stability_Score'].apply(categorize_stability)
        
        return df
    
    @staticmethod
    def create_target_encodings(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Create target encoding for high-cardinality features
        """
        df = df.copy()
        
        # Calculate mean target by CITY
        if 'CITY' in df.columns and target_col in df.columns:
            city_risk = df.groupby('CITY')[target_col].mean().to_dict()
            df['City_Risk_Encoding'] = df['CITY'].map(city_risk)
        
        # Calculate mean target by STATE
        if 'STATE' in df.columns and target_col in df.columns:
            state_risk = df.groupby('STATE')[target_col].mean().to_dict()
            df['State_Risk_Encoding'] = df['STATE'].map(state_risk)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        """
        df = self.create_interaction_features(df)
        df = self.create_categorical_features(df)
        
        if target_col and target_col in df.columns:
            df = self.create_target_encodings(df, target_col)
        
        return df
    
    def get_feature_types(self) -> Dict[str, list]:
        """
        Get lists of different feature types
        """
        return {
            'numerical': NUMERICAL_FEATURES + [
                'Income_Age_Ratio', 'Experience_Age_Ratio',
                'Stability_Score', 'DTI_Ratio'
            ],
            'categorical': CATEGORICAL_FEATURES + [
                'Age_Group', 'Income_Category', 'Stability_Category'
            ],
            'target_encoded': ['City_Risk_Encoding', 'State_Risk_Encoding']
        }
