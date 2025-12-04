"""
Configuration settings for the loan risk predictor
"""

# Feature configurations
NUMERICAL_FEATURES = [
    'Income', 'Age', 'Experience', 
    'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS'
]

CATEGORICAL_FEATURES = [
    'Married/Single', 'House_Ownership', 'Car_Ownership',
    'Profession', 'CITY', 'STATE'
]

ENGINEERED_FEATURES = [
    'Income_Age_Ratio', 'Experience_Age_Ratio',
    'Stability_Score', 'DTI_Ratio',
    'Age_Group', 'Income_Category', 'Stability_Category'
]

TARGET_COLUMN = 'Risk_Flag'

# Model configuration
MODEL_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}

# Threshold configuration
DEFAULT_THRESHOLD = 0.5
RISK_LEVELS = {
    'LOW': (0, 0.3),
    'MEDIUM': (0.3, 0.7),
    'HIGH': (0.7, 1.0)
}

# Path configuration
DEFAULT_MODEL_PATH = 'model_artifacts/model.pkl'
DEFAULT_PREPROCESSOR_PATH = 'model_artifacts/preprocessor.pkl'
