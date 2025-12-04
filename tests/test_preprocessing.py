import pytest
import pandas as pd
import numpy as np
from src.loan_risk_predictor.preprocessing import DataPreprocessor

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data = {
        'Income': [1000000, 2000000, 3000000],
        'Age': [30, 40, 50],
        'Experience': [5, 10, 15],
        'Married/Single': ['single', 'married', 'single'],
        'House_Ownership': ['rented', 'owned', 'rented'],
        'Car_Ownership': ['yes', 'no', 'yes'],
        'Profession': ['Engineer', 'Doctor', 'Teacher'],
        'CITY': ['CityA', 'CityB', 'CityC'],
        'STATE': ['StateA', 'StateB', 'StateA'],
        'CURRENT_JOB_YRS': [2, 5, 10],
        'CURRENT_HOUSE_YRS': [1, 3, 5],
        'Risk_Flag': [0, 1, 0]
    }
    return pd.DataFrame(data)

def test_preprocessor_creation():
    """Test preprocessor creation"""
    preprocessor = DataPreprocessor()
    assert preprocessor.preprocessor is None
    assert not preprocessor.is_fitted

def test_fit_transform(sample_data):
    """Test fit_transform method"""
    preprocessor = DataPreprocessor()
    X = sample_data.drop(columns=['Risk_Flag'])
    y = sample_data['Risk_Flag']
    
    transformed = preprocessor.fit_transform(X, y)
    
    assert preprocessor.is_fitted
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(X)

def test_transform_without_fit(sample_data):
    """Test transform without fitting raises error"""
    preprocessor = DataPreprocessor()
    X = sample_data.drop(columns=['Risk_Flag'])
    
    with pytest.raises(ValueError):
        preprocessor.transform(X)

def test_feature_names(sample_data):
    """Test get_feature_names method"""
    preprocessor = DataPreprocessor()
    X = sample_data.drop(columns=['Risk_Flag'])
    y = sample_data['Risk_Flag']
    
    preprocessor.fit(X, y)
    feature_names = preprocessor.get_feature_names()
    
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
