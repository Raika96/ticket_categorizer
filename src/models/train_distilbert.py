"""Main Model: DistilBERT Fine-tuning for Text Classification"""

import pandas as pd
import numpy as np
import json
import torch
import os
from datetime import datetime
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns


class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }


class DistilBERTClassifier:
    def __init__(self, num_labels=9, confidence_threshold=0.6, max_length=128):
        self.num_labels = num_labels
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = None
        self.label_mapping = None
        self.training_stats = {}
    
    def load_data(self, train_path, val_path, test_path, mapping_path):
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)
        with open(mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
        
        print(f"Loaded: {len(self.train_df)} train, {len(self.val_df)} val, {len(self.test_df)} test")
        
        self.train_dataset = TicketDataset(
            self.train_df['text_clean'].values, self.train_df['label'].values,
            self.tokenizer, self.max_length
        )
        self.val_dataset = TicketDataset(
            self.val_df['text_clean'].values, self.val_df['label'].values,
            self.tokenizer, self.max_length
        )
        self.test_dataset = TicketDataset(
            self.test_df['text_clean'].values, self.test_df['label'].values,
            self.tokenizer, self.max_length
        )
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='weighted'),
            'precision': precision_score(labels, preds, average='weighted', zero_division=0),
            'recall': recall_score(labels, preds, average='weighted', zero_division=0)
        }
    
    def train(self, output_dir='models/distilbert', num_epochs=3, batch_size=16, learning_rate=2e-5, early_stopping_patience=3):
        print(f"\nTraining DistilBERT | Epochs: {num_epochs} | Batch: {batch_size} | LR: {learning_rate}")
        
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=self.num_labels
        )
        self.model.to(self.device)
        
        class_counts = self.train_df['label'].value_counts().sort_index().values
        class_weights = len(self.train_df) / (self.num_labels * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        print(f"   Using class weights for imbalanced data")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=100,
            eval_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            warmup_steps=500,
            report_to='none'
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )
        
        start_time = datetime.now()
        train_result = trainer.train()
        training_time = (datetime.now() - start_time).total_seconds()
        
        print(f"Training complete in {training_time:.1f}s ({training_time/60:.1f} min)")
        
        self.training_stats = {
            'training_time_seconds': training_time,
            'train_loss': float(train_result.training_loss)
        }
        return self
    
    def evaluate(self, split='val', save_plots=True):
        print(f"\nEvaluating on {split.upper()} set...")
        dataset = self.val_dataset if split == 'val' else self.test_dataset
        df = self.val_df if split == 'val' else self.test_df
        
        self.model.eval()
        all_preds, all_probs, all_labels = [], [], []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                batch = dataset[i]
                input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                
                all_preds.append(torch.argmax(probs, dim=1).cpu().numpy()[0])
                all_probs.append(probs.cpu().numpy()[0])
                all_labels.append(batch['labels'].numpy())
        
        y_pred = np.array(all_preds)
        y_proba = np.array(all_probs)
        y_true = np.array(all_labels)
        
        n_uncategorized = (y_proba.max(axis=1) < self.confidence_threshold).sum()
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"   Accuracy: {acc:.4f} | F1: {f1:.4f}")
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
            'f1_weighted': float(f1),
            'precision_weighted': float(precision_score(y_true, y_pred, average='weighted')),
            'recall_weighted': float(recall_score(y_true, y_pred, average='weighted')),
            'n_uncategorized': int(n_uncategorized),
            'uncategorized_percentage': float(n_uncategorized/len(df)*100)
        }
    
    def _plot_confusion_matrix(self, y_true, y_pred, split):
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Purples',
                   xticklabels=[self.label_mapping['id_to_category'][str(i)][:15] 
                               for i in range(self.label_mapping['num_classes'])],
                   yticklabels=[self.label_mapping['id_to_category'][str(i)][:15] 
                               for i in range(self.label_mapping['num_classes'])])
        
        plt.title(f'Confusion Matrix - DistilBERT ({split.upper()})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        os.makedirs('models/plots', exist_ok=True)
        plot_path = f'models/plots/distilbert_cm_{split}.png'
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {plot_path}")
    
    def predict(self, texts, return_confidence=True):
        self.model.eval()
        predictions, confidences = [], []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text, max_length=self.max_length, padding='max_length',
                    truncation=True, return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1)
                confidence = probs.max().item()
                
                pred_label = self.label_mapping['id_to_category'][str(pred.item())]
                pred_label = 'Uncategorized' if confidence < self.confidence_threshold else pred_label
                
                predictions.append(pred_label)
                confidences.append(confidence)
        
        return (predictions, confidences) if return_confidence else predictions
    
    def save(self, model_dir='models/distilbert/final'):
        os.makedirs(model_dir, exist_ok=True)
        
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        config = {
            'model_type': 'distilbert_sequence_classification',
            'confidence_threshold': self.confidence_threshold,
            'max_length': self.max_length,
            'num_labels': self.num_labels,
            'training_stats': self.training_stats,
            'label_mapping': self.label_mapping,
            'created_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(model_dir, 'classifier_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Model saved to: {model_dir}")
    
    @classmethod
    def load(cls, model_dir='models/distilbert/final'):
        with open(os.path.join(model_dir, 'classifier_config.json'), 'r') as f:
            config = json.load(f)
        
        instance = cls(
            num_labels=config['num_labels'],
            confidence_threshold=config['confidence_threshold'],
            max_length=config['max_length']
        )
        
        instance.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        instance.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        instance.model.to(instance.device)
        instance.label_mapping = config['label_mapping']
        instance.training_stats = config['training_stats']
        return instance


if __name__ == "__main__":
    classifier = DistilBERTClassifier(num_labels=9, confidence_threshold=0.6)
    classifier.load_data(
        train_path='data/processed/train.csv',
        val_path='data/processed/val.csv',
        test_path='data/processed/test.csv',
        mapping_path='data/processed/label_mapping.json'
    )
    classifier.train(output_dir='models/distilbert/checkpoints', num_epochs=3)
    val_metrics = classifier.evaluate(split='val', save_plots=True)
    classifier.save(model_dir='models/distilbert')
