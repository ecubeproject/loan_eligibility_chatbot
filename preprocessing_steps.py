from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class CleanAndPreprocess(BaseEstimator, TransformerMixin):
    """Remove duplicates, impute missing values, and drop irrelevant columns."""
    def fit(self, df: pd.DataFrame, y=None):
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        
        # Impute missing values
        median_imputer = SimpleImputer(strategy='median')
        mode_imputer = SimpleImputer(strategy='most_frequent')
        
        # Impute numerical and categorical columns separately
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        df[numerical_cols] = median_imputer.fit_transform(df[numerical_cols])
        df[categorical_cols] = mode_imputer.fit_transform(df[categorical_cols])
        
        # Drop Loan_ID column if exists
        if 'Loan_ID' in df.columns:
            df = df.drop('Loan_ID', axis=1)
        
        return df

class DynamicEncodeVariables(BaseEstimator, TransformerMixin):
    """Encode binary and multi-category categorical columns."""
    def fit(self, df: pd.DataFrame, y=None):
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        
        # Define mappings explicitly to ensure correct encoding
        encoding_mappings = {
            'Gender': {'Male': 0, 'Female': 1},
            'Married': {'No': 0, 'Yes': 1},
            'Education': {'Not Graduate': 0, 'Graduate': 1},
            'Self_Employed': {'No': 0, 'Yes': 1},
            'Property_Area': {'Rural': 1, 'Urban': 0, 'Semiurban': 0}
        }
        
        # Apply mappings to categorical columns
        for column, mapping in encoding_mappings.items():
            if column in df_encoded.columns:
                df_encoded[column] = df_encoded[column].map(mapping)
        
        # One-hot encoding for multi-category variables (like Property_Area if still needed)
        multi_category_cols = [col for col in df_encoded.columns if df_encoded[col].dtype == 'object']
        df_encoded = pd.get_dummies(df_encoded, columns=multi_category_cols, drop_first=True)
        
        # Ensure all expected transformations are applied
        return df_encoded

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Apply log transformation, create new features, and drop unused columns."""
    def __init__(self, skewed_columns: list):
        self.skewed_columns = skewed_columns

    def fit(self, df: pd.DataFrame, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Log transformation
        for col in self.skewed_columns:
            df[col] = np.log1p(df[col])
        
        # Create Family_Income
        df['Family_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        
        # Simplify Dependents
        def simplify_dependents(row):
            if row.get('Dependents_1', 0) == 1:
                return 1
            elif row.get('Dependents_2', 0) == 2:
                return 2
            elif row.get('Dependents_3+', 0) == 3:
                return 3
            else:
                return 0

        df['Dependents'] = df.apply(simplify_dependents, axis=1)
        
        # Create Is_Rural indicator, ensure result is a Series for consistent handling
        is_semiurban = pd.Series(df.get('Property_Area_Semiurban', 0) == 0)
        is_urban = pd.Series(df.get('Property_Area_Urban', 0) == 0)
        df['Is_Rural'] = (is_semiurban & is_urban).astype(int)
        
        # Drop old columns
        columns_to_drop = ['Property_Area_Semiurban', 'Property_Area_Urban', 'Dependents_1', 'Dependents_2', 'Dependents_3+', 'ApplicantIncome', 'CoapplicantIncome']
        df = df.drop(columns=columns_to_drop, errors='ignore')
        
        return df

# Create the complete preprocessing pipeline
def create_preprocessing_pipeline(skewed_columns: list) -> Pipeline:
    """
    Creates a preprocessing pipeline for transforming raw data into the format
    required for model prediction, excluding the balancing step.
    
    Parameters:
    skewed_columns (list): List of column names for log transformation.

    Returns:
    Pipeline: A Scikit-Learn Pipeline for preprocessing.
    """
    pipeline = Pipeline([
        ('cleaning', CleanAndPreprocess()),
        ('encoding', DynamicEncodeVariables()),
        ('feature_engineering', FeatureEngineering(skewed_columns=skewed_columns)),
    ])
    return pipeline
