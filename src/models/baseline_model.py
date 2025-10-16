"""Baseline Model: TF-IDF + Logistic Regression"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, precision_score, recall_score)
import matplotlib.pyplot as plt
import seaborn as sns


class BaselineTicketClassifier:
    def __init__(self, confidence_threshold=0.6):
        self.confidence_threshold = confidence_threshold
        self.vectorizer = TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95,
            strip_accents='unicode', lowercase=True, stop_words='english'
        )
        self.model = LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1
        )
        self.label_mapping = None
        self.training_stats = {}
    
    def load_data(self, train_path, val_path, test_path, mapping_path):
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)
        with open(mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
        print(f"Loaded: {len(self.train_df)} train, {len(self.val_df)} val, {len(self.test_df)} test")
    
    def train(self):
        print("\nTraining Baseline Model...")
        X_train = self.train_df['text_clean'].values
        y_train = self.train_df['label'].values
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"   TF-IDF vocabulary: {len(self.vectorizer.vocabulary_):,} words")
        
        start_time = datetime.now()
        self.model.fit(X_train_tfidf, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        train_pred = self.model.predict(X_train_tfidf)
        train_acc = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        
        print(f"Training complete in {training_time:.2f}s | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        
        self.training_stats = {
            'training_time_seconds': training_time,
            'train_accuracy': float(train_acc),
            'train_f1_weighted': float(train_f1)
        }
        return self
    
    def evaluate(self, split='val', save_plots=True):
        print(f"\nEvaluating on {split.upper()} set...")
        df = self.val_df if split == 'val' else self.test_df
        X = df['text_clean'].values
        y_true = df['label'].values
        
        X_tfidf = self.vectorizer.transform(X)
        y_pred = self.model.predict(X_tfidf)
        y_proba = self.model.predict_proba(X_tfidf)
        
        max_proba = y_proba.max(axis=1)
        n_uncategorized = (max_proba < self.confidence_threshold).sum()
        
        acc = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        
        print(f"   Accuracy: {acc:.4f} | F1: {f1_weighted:.4f}")
        print(f"   Uncategorized: {n_uncategorized}/{len(df)} ({n_uncategorized/len(df)*100:.1f}%)")
        
        report = classification_report(
            y_true, y_pred,
            target_names=[self.label_mapping['id_to_category'][str(i)] 
                         for i in range(self.label_mapping['num_classes'])],
            digits=4
        )
        print(f"\n{report}")
        
        if save_plots:
            self._plot_confusion_matrix(y_true, y_pred, split)
        
        return {
            'split': split,
            'accuracy': float(acc),
            'f1_weighted': float(f1_weighted),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'n_uncategorized': int(n_uncategorized),
            'uncategorized_percentage': float(n_uncategorized/len(df)*100)
        }
    
    def _plot_confusion_matrix(self, y_true, y_pred, split):
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=[self.label_mapping['id_to_category'][str(i)][:15] 
                               for i in range(self.label_mapping['num_classes'])],
                   yticklabels=[self.label_mapping['id_to_category'][str(i)][:15] 
                               for i in range(self.label_mapping['num_classes'])])
        
        plt.title(f'Confusion Matrix - Baseline ({split.upper()})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        os.makedirs('models/plots', exist_ok=True)
        plot_path = f'models/plots/baseline_cm_{split}.png'
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {plot_path}")
    
    def predict(self, texts, return_confidence=True):
        X_tfidf = self.vectorizer.transform(texts)
        y_pred = self.model.predict(X_tfidf)
        y_proba = self.model.predict_proba(X_tfidf)
        confidences = y_proba.max(axis=1)
        
        predictions = [self.label_mapping['id_to_category'][str(pred)] for pred in y_pred]
        predictions = ['Uncategorized' if conf < self.confidence_threshold else pred
                      for pred, conf in zip(predictions, confidences)]
        
        return (predictions, confidences) if return_confidence else predictions
    
    def save(self, model_dir='models/baseline'):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(model_dir, 'vectorizer.joblib'))
        joblib.dump(self.model, os.path.join(model_dir, 'model.joblib'))
        
        config = {
            'model_type': 'baseline_tfidf_logistic_regression',
            'confidence_threshold': self.confidence_threshold,
            'training_stats': self.training_stats,
            'label_mapping': self.label_mapping,
            'created_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Model saved to: {model_dir}")
    
    @classmethod
    def load(cls, model_dir='models/baseline'):
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        instance = cls(confidence_threshold=config['confidence_threshold'])
        instance.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.joblib'))
        instance.model = joblib.load(os.path.join(model_dir, 'model.joblib'))
        instance.label_mapping = config['label_mapping']
        instance.training_stats = config['training_stats']
        return instance


if __name__ == "__main__":
    classifier = BaselineTicketClassifier(confidence_threshold=0.6)
    classifier.load_data(
        train_path='data/processed/train.csv',
        val_path='data/processed/val.csv',
        test_path='data/processed/test.csv',
        mapping_path='data/processed/label_mapping.json'
    )
    classifier.train()
    val_metrics = classifier.evaluate(split='val', save_plots=True)
    classifier.save(model_dir='models/baseline')
