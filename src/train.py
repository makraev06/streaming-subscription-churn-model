import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from src.preprocessing import FeatureEngineeringTransformer, get_preprocessor

def train_and_export_model():
    print("Memuat dataset train.csv...")
    data_path = 'train.csv'
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['churned'])
    y = df['churned']
    
    print(f"Data shape: {X.shape}, Target distribution:\n{y.value_counts(normalize=True)}")
    
    # Train-val split for evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Complete End-to-End Pipeline
    pipeline = Pipeline(steps=[
        ('feature_engineering', FeatureEngineeringTransformer()),
        ('preprocessor', get_preprocessor()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("Melatih End-to-End Pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Validation evaluation
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    
    metrics = {
        'roc_auc': float(roc_auc_score(y_val, val_probs)),
        'f1': float(f1_score(y_val, val_preds)),
        'accuracy': float(accuracy_score(y_val, val_preds)),
        'precision': float(precision_score(y_val, val_preds)),
        'recall': float(recall_score(y_val, val_preds))
    }
    
    print(f"Hasil Evaluasi Validasi:\nROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    
    # Fit full pipeline on all train data for maximum deployment performance
    print("Melatih Pipeline Akhir pada Seluruh Data Pelatihan...")
    pipeline.fit(X, y)
    
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'churn_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"Model berhasil disimpan di: {model_path}")
    
    # Save metrics metadata for Streamlit
    joblib.dump(metrics, os.path.join('models', 'metrics.pkl'))
    print("Metrik evaluasi berhasil disimpan!")

if __name__ == '__main__':
    train_and_export_model()
