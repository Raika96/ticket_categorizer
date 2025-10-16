"""
Model Evaluation & Error Analysis
Comprehensive evaluation with multi-class metrics, confusion matrix, and error analysis

Usage:
    python evaluate.py --model_dir models --output_dir experiments
    python evaluate.py --use_test_set  # Final evaluation (run once only!)
"""

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score
)
from collections import Counter, defaultdict


def load_model(model_dir, model_type='baseline'):
    """Load trained model"""
    print(f"\n{'='*70}")
    print(f"📦 Loading {model_type.upper()} model from {model_dir}...")
    print(f"{'='*70}")
    
    if model_type == 'baseline':
        vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.joblib'))
        model = joblib.load(os.path.join(model_dir, 'model.joblib'))
        
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        print("✅ Baseline model loaded successfully")
        return model, vectorizer, config
    
    elif model_type == 'distilbert':
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        import torch
        
        # Load model and tokenizer directly from model_dir
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        
        # Load classifier config
        config_path = os.path.join(model_dir, 'classifier_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("✅ DistilBERT model loaded successfully")
        return model, tokenizer, config
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_data(data_path, label_mapping_path):
    """Load evaluation data"""
    df = pd.read_csv(data_path)
    
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    
    return df, label_mapping


def compute_comprehensive_metrics(y_true, y_pred, y_proba, label_mapping, confidence_threshold=0.6):
    """Compute all evaluation metrics"""
    print("\n" + "="*70)
    print(" COMPREHENSIVE METRICS")
    print("="*70)
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    
    # Cohen's Kappa (inter-rater agreement)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Matthews Correlation Coefficient
    try:
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    except:
        metrics['matthews_corrcoef'] = None
    
    # Confidence analysis
    max_proba = y_proba.max(axis=1)
    metrics['mean_confidence'] = float(np.mean(max_proba))
    metrics['median_confidence'] = float(np.median(max_proba))
    metrics['confidence_std'] = float(np.std(max_proba))
    
    # Confidence threshold analysis
    high_confidence_mask = max_proba >= confidence_threshold
    metrics['n_high_confidence'] = int(high_confidence_mask.sum())
    metrics['n_low_confidence'] = int((~high_confidence_mask).sum())
    metrics['high_confidence_rate'] = float(high_confidence_mask.sum() / len(y_true))
    
    # Accuracy by confidence
    if metrics['n_high_confidence'] > 0:
        metrics['accuracy_high_confidence'] = accuracy_score(
            y_true[high_confidence_mask], 
            y_pred[high_confidence_mask]
        )
    else:
        metrics['accuracy_high_confidence'] = None
    
    if metrics['n_low_confidence'] > 0:
        metrics['accuracy_low_confidence'] = accuracy_score(
            y_true[~high_confidence_mask], 
            y_pred[~high_confidence_mask]
        )
    else:
        metrics['accuracy_low_confidence'] = None
    
    # Print summary
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"   F1-Score (weighted): {metrics['f1_weighted']:.4f}")
    print(f"   F1-Score (macro): {metrics['f1_macro']:.4f}")
    print(f"   Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    
    print(f"\n🎯 Confidence Analysis:")
    print(f"   Mean confidence: {metrics['mean_confidence']:.4f}")
    print(f"   High confidence (≥{confidence_threshold}): {metrics['n_high_confidence']} ({metrics['high_confidence_rate']*100:.2f}%)")
    print(f"   Low confidence (<{confidence_threshold}): {metrics['n_low_confidence']} ({(1-metrics['high_confidence_rate'])*100:.2f}%)")
    if metrics['accuracy_high_confidence']:
        print(f"   Accuracy (high conf): {metrics['accuracy_high_confidence']:.4f}")
    if metrics['accuracy_low_confidence']:
        print(f"   Accuracy (low conf): {metrics['accuracy_low_confidence']:.4f}")
    
    return metrics


def generate_classification_report(y_true, y_pred, label_mapping):
    """Generate detailed per-class report"""
    print("\n" + "="*70)
    print(" PER-CLASS PERFORMANCE")
    print("="*70)
    
    target_names = [label_mapping['id_to_category'][str(i)] 
                   for i in range(label_mapping['num_classes'])]
    
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print(report)
    
    # Also get as dict for saving
    report_dict = classification_report(y_true, y_pred, target_names=target_names, 
                                       output_dict=True, zero_division=0)
    
    return report, report_dict


def plot_confusion_matrix(y_true, y_pred, label_mapping, output_dir, split_name='val'):
    """Plot detailed confusion matrix"""
    print(f"\n📊 Generating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    target_names = [label_mapping['id_to_category'][str(i)][:25] 
                   for i in range(label_mapping['num_classes'])]
    
    # Plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
               xticklabels=target_names, yticklabels=target_names)
    ax1.set_title(f'Confusion Matrix - Counts ({split_name.upper()} Set)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Category', fontsize=12)
    ax1.set_ylabel('True Category', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Normalized (percentages)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Oranges', ax=ax2,
               xticklabels=target_names, yticklabels=target_names)
    ax2.set_title(f'Confusion Matrix - Normalized ({split_name.upper()} Set)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Category', fontsize=12)
    ax2.set_ylabel('True Category', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'confusion_matrix_{split_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    return cm, cm_normalized


def analyze_errors(df, y_true, y_pred, y_proba, label_mapping, output_dir, n_examples=20):
    """Perform detailed error analysis"""
    print("\n" + "="*70)
    print(" ERROR ANALYSIS")
    print("="*70)
    
    # Find misclassifications
    misclassified = y_true != y_pred
    n_errors = misclassified.sum()
    error_rate = n_errors / len(y_true)
    
    print(f"\n❌ Total Errors: {n_errors} / {len(y_true)} ({error_rate*100:.2f}%)")
    
    error_analysis = {}
    
    # 1. Most common misclassification patterns
    print(f"\n🔍 Most Common Misclassification Patterns:")
    misclass_patterns = []
    for true_label, pred_label in zip(y_true[misclassified], y_pred[misclassified]):
        true_cat = label_mapping['id_to_category'][str(true_label)]
        pred_cat = label_mapping['id_to_category'][str(pred_label)]
        misclass_patterns.append((true_cat, pred_cat))
    
    pattern_counts = Counter(misclass_patterns)
    top_patterns = pattern_counts.most_common(10)
    
    error_analysis['top_misclassification_patterns'] = []
    for (true_cat, pred_cat), count in top_patterns:
        pct = count / n_errors * 100
        print(f"   {true_cat[:30]:30} → {pred_cat[:30]:30}: {count:4} ({pct:5.2f}%)")
        error_analysis['top_misclassification_patterns'].append({
            'true_category': true_cat,
            'predicted_category': pred_cat,
            'count': int(count),
            'percentage': float(pct)
        })
    
    # 2. Error rate by category
    print(f"\n📊 Error Rate by Category:")
    category_errors = defaultdict(lambda: {'total': 0, 'errors': 0})
    
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        cat_name = label_mapping['id_to_category'][str(true_label)]
        category_errors[cat_name]['total'] += 1
        if true_label != pred_label:
            category_errors[cat_name]['errors'] += 1
    
    error_analysis['category_error_rates'] = []
    for cat_name in sorted(category_errors.keys()):
        stats = category_errors[cat_name]
        error_rate = stats['errors'] / stats['total']
        print(f"   {cat_name[:40]:40}: {stats['errors']:4}/{stats['total']:4} ({error_rate*100:5.2f}%)")
        error_analysis['category_error_rates'].append({
            'category': cat_name,
            'errors': stats['errors'],
            'total': stats['total'],
            'error_rate': float(error_rate)
        })
    
    # 3. Confidence analysis of errors
    error_confidences = y_proba.max(axis=1)[misclassified]
    correct_confidences = y_proba.max(axis=1)[~misclassified]
    
    print(f"\n🎯 Confidence Distribution:")
    print(f"   Errors - Mean confidence: {error_confidences.mean():.4f} (±{error_confidences.std():.4f})")
    print(f"   Correct - Mean confidence: {correct_confidences.mean():.4f} (±{correct_confidences.std():.4f})")
    
    error_analysis['confidence_analysis'] = {
        'errors_mean_confidence': float(error_confidences.mean()),
        'errors_std_confidence': float(error_confidences.std()),
        'correct_mean_confidence': float(correct_confidences.mean()),
        'correct_std_confidence': float(correct_confidences.std())
    }
    
    # 4. Save example errors
    print(f"\n📝 Saving {min(n_examples, n_errors)} Error Examples...")
    
    error_df = df[misclassified].copy()
    error_df['true_label'] = y_true[misclassified]
    error_df['predicted_label'] = y_pred[misclassified]
    error_df['confidence'] = y_proba.max(axis=1)[misclassified]
    error_df['true_category'] = error_df['true_label'].apply(
        lambda x: label_mapping['id_to_category'][str(x)]
    )
    error_df['predicted_category'] = error_df['predicted_label'].apply(
        lambda x: label_mapping['id_to_category'][str(x)]
    )
    
    # Sort by confidence (most confident mistakes are most interesting)
    error_df = error_df.sort_values('confidence', ascending=False)
    
    # Select columns for export
    export_cols = ['text_clean', 'true_category', 'predicted_category', 'confidence']
    error_examples = error_df[export_cols].head(n_examples)
    
    error_file = os.path.join(output_dir, 'error_examples.csv')
    error_examples.to_csv(error_file, index=False)
    print(f"✅ Saved error examples: {error_file}")
    
    # 5. Plot confidence distributions
    plot_confidence_analysis(correct_confidences, error_confidences, output_dir)
    
    return error_analysis


def plot_confidence_analysis(correct_conf, error_conf, output_dir):
    """Plot confidence distribution for correct vs incorrect predictions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(correct_conf, bins=50, alpha=0.7, label='Correct', color='green', density=True)
    ax1.hist(error_conf, bins=50, alpha=0.7, label='Errors', color='red', density=True)
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Confidence Distribution: Correct vs Errors', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Box plot
    ax2.boxplot([correct_conf, error_conf], labels=['Correct', 'Errors'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.set_ylabel('Confidence', fontsize=12)
    ax2.set_title('Confidence Box Plot', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'confidence_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved confidence analysis: {output_path}")
    plt.close()


def generate_summary_report(metrics, error_analysis, output_dir):
    """Generate text summary report"""
    report_path = os.path.join(output_dir, 'evaluation_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(" MODEL EVALUATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"  F1-Score (weighted): {metrics['f1_weighted']:.4f}\n")
        f.write(f"  F1-Score (macro): {metrics['f1_macro']:.4f}\n")
        f.write(f"  Precision (weighted): {metrics['precision_weighted']:.4f}\n")
        f.write(f"  Recall (weighted): {metrics['recall_weighted']:.4f}\n")
        f.write(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}\n\n")
        
        f.write("CONFIDENCE ANALYSIS:\n")
        f.write(f"  Mean confidence: {metrics['mean_confidence']:.4f}\n")
        f.write(f"  High confidence rate: {metrics['high_confidence_rate']*100:.2f}%\n")
        if metrics['accuracy_high_confidence']:
            f.write(f"  Accuracy (high conf): {metrics['accuracy_high_confidence']:.4f}\n")
        if metrics['accuracy_low_confidence']:
            f.write(f"  Accuracy (low conf): {metrics['accuracy_low_confidence']:.4f}\n")
        f.write("\n")
        
        f.write("TOP MISCLASSIFICATION PATTERNS:\n")
        for pattern in error_analysis['top_misclassification_patterns'][:5]:
            f.write(f"  {pattern['true_category']} → {pattern['predicted_category']}: "
                   f"{pattern['count']} ({pattern['percentage']:.2f}%)\n")
        f.write("\n")
        
        f.write("KEY INSIGHTS:\n")
        f.write(f"  - Model correctly classifies {metrics['accuracy']*100:.2f}% of tickets\n")
        f.write(f"  - {metrics['high_confidence_rate']*100:.2f}% can be auto-assigned with high confidence\n")
        f.write(f"  - Errors have {error_analysis['confidence_analysis']['errors_mean_confidence']:.2f} "
               f"avg confidence (vs {error_analysis['confidence_analysis']['correct_mean_confidence']:.2f} for correct)\n")
        f.write(f"  - Strong agreement between model and ground truth (Kappa: {metrics['cohen_kappa']:.4f})\n")
    
    print(f"\n✅ Saved evaluation summary: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Model and Analyze Errors')
    parser.add_argument('--model_type', type=str, default='baseline', 
                       choices=['baseline', 'distilbert'],
                       help='Model type to evaluate')
    parser.add_argument('--model_dir', type=str, default='models', help='Model directory')
    parser.add_argument('--data_path', type=str, default='data/processed/val.csv', 
                       help='Evaluation data path')
    parser.add_argument('--label_mapping', type=str, default='data/processed/label_mapping.json',
                       help='Label mapping file')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory for evaluation results')
    parser.add_argument('--use_test_set', action='store_true',
                       help='Use test set (FINAL EVALUATION - run once only!)')
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Override data path if using test set
    if args.use_test_set:
        args.data_path = 'data/processed/test.csv'
        print("\n⚠️  WARNING: Using TEST SET for final evaluation!")
        print("This should only be done ONCE after all development is complete.\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print(" MODEL EVALUATION & ERROR ANALYSIS")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model_dir}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    
    # Load model and data
    import time
    start_time = time.time()
    
    model, preprocessor, config = load_model(args.model_dir, args.model_type)
    
    print(f"\n{'='*70}")
    print(f"📊 Loading data from {args.data_path}...")
    print(f"{'='*70}")
    df, label_mapping = load_data(args.data_path, args.label_mapping)
    print(f"✅ Data loaded: {len(df)} samples, {len(label_mapping['id_to_category'])} classes")
    
    split_name = 'test' if args.use_test_set else 'val'
    print(f"\n{'='*70}")
    print(f"🔍 Evaluating on {split_name.upper()} set: {len(df)} samples")
    print(f"{'='*70}")
    
    # Transform and predict
    print("\n⚙️  Making predictions...")
    
    if args.model_type == 'baseline':
        # TF-IDF + Logistic Regression
        X = preprocessor.transform(df['text_clean'].values)
        y_true = df['label'].values
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
    elif args.model_type == 'distilbert':
        # DistilBERT
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm import tqdm
        
        tokenizer = preprocessor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {device}")
        model.to(device)
        model.eval()
        
        # Tokenize
        print(f"📝 Tokenizing {len(df)} samples...")
        encodings = tokenizer(
            df['text_clean'].tolist(),
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        print(f"✅ Tokenization complete")
        
        # Create dataset and dataloader
        dataset = TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask']
        )
        batch_size = 32
        dataloader = DataLoader(dataset, batch_size=batch_size)
        print(f"📦 Created dataloader with batch_size={batch_size} ({len(dataloader)} batches)")
        
        # Predict
        all_preds = []
        all_probs = []
        
        print(f"🔄 Running inference...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
                input_ids, attention_mask = [b.to(device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Show progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    processed = min((batch_idx + 1) * batch_size, len(df))
                    print(f"   Processed {processed}/{len(df)} samples ({processed/len(df)*100:.1f}%)", end='\r')
        
        print()  # New line after progress
        y_true = df['label'].values
        y_pred = np.array(all_preds)
        y_proba = np.array(all_probs)
        print(f"✅ Inference complete! Processed {len(y_pred)} samples")
    
    print("\n✅ Predictions complete")
    
    # Compute metrics
    metrics = compute_comprehensive_metrics(
        y_true, y_pred, y_proba, label_mapping, args.confidence_threshold
    )
    
    # Classification report
    report_str, report_dict = generate_classification_report(y_true, y_pred, label_mapping)
    
    # Confusion matrix
    cm, cm_norm = plot_confusion_matrix(y_true, y_pred, label_mapping, args.output_dir, split_name)
    
    # Error analysis
    error_analysis = analyze_errors(df, y_true, y_pred, y_proba, label_mapping, 
                                   args.output_dir, n_examples=50)
    
    # Generate summary
    generate_summary_report(metrics, error_analysis, args.output_dir)
    
    # Save all results
    results = {
        'timestamp': datetime.now().isoformat(),
        'split': split_name,
        'n_samples': len(df),
        'overall_metrics': metrics,
        'per_class_metrics': report_dict,
        'error_analysis': error_analysis
    }
    
    results_file = os.path.join(args.output_dir, f'evaluation_results_{split_name}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved complete results: {results_file}")
    
    # Save classification report
    report_file = os.path.join(args.output_dir, f'classification_report_{split_name}.txt')
    with open(report_file, 'w') as f:
        f.write(report_str)
    print(f"✅ Saved classification report: {report_file}")
    
    print("\n" + "="*70)
    print(" EVALUATION COMPLETE ✅")
    print("="*70)
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Total evaluation time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    
    print(f"\n📊 Key Results:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   F1-Score: {metrics['f1_weighted']:.4f}")
    print(f"   Auto-assignment Rate: {metrics['high_confidence_rate']*100:.2f}%")
    print(f"\n📁 All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

