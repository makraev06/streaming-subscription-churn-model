import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer for streaming churn feature engineering:
    - tenure_days: abs(signup_date)
    - listening_intensity: weekly_songs_played / (average_session_length + 1e-6)
    - skip_behavior: categorical binning of song_skip_rate
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # 1. tenure_days
        if 'signup_date' in X_out.columns:
            X_out['tenure_days'] = X_out['signup_date'].abs()
            X_out = X_out.drop(columns=['signup_date'])
        
        # 2. listening_intensity
        if 'weekly_songs_played' in X_out.columns and 'average_session_length' in X_out.columns:
            session_len = X_out['average_session_length'].replace(0, np.nan)
            X_out['listening_intensity'] = X_out['weekly_songs_played'] / (session_len + 1e-6)
            X_out['listening_intensity'] = X_out['listening_intensity'].fillna(0)
        
        # 3. skip_behavior
        if 'song_skip_rate' in X_out.columns:
            X_out['skip_behavior'] = pd.cut(
                X_out['song_skip_rate'],
                bins=[-np.inf, 0.33, 0.66, np.inf],
                labels=['Low', 'Medium', 'High']
            ).astype(str)
            
        # 4. drop customer_id if present
        if 'customer_id' in X_out.columns:
            X_out = X_out.drop(columns=['customer_id'])
            
        return X_out

def get_preprocessor():
    numeric_features = [
        'age',
        'weekly_hours',
        'average_session_length',
        'song_skip_rate',
        'weekly_songs_played',
        'weekly_unique_songs',
        'num_favorite_artists',
        'num_platform_friends',
        'num_playlists_created',
        'num_shared_playlists',
        'num_subscription_pauses',
        'notifications_clicked',
        'tenure_days',
        'listening_intensity'
    ]

    categorical_features = [
        'location',
        'subscription_type',
        'payment_plan',
        'payment_method',
        'customer_service_inquiries',
        'skip_behavior'
    ]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor

def get_risk_tier(probability):
    if probability >= 0.70:
        return 'High', '🔴', 'Tinggi'
    elif probability >= 0.30:
        return 'Medium', '🟡', 'Sedang'
    else:
        return 'Low', '🟢', 'Rendah'
