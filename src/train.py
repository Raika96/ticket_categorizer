"""Training Script for Support Ticket Classification
Usage:
    python src/train.py --model baseline
    python src/train.py --model distilbert --epochs 3 --batch_size 16
"""

import argparse
import json
import os
import sys
from datetime import datetime
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def train_baseline(args):
    from src.models.baseline_model import BaselineTicketClassifier
    
    print(f"\n{'='*70}")
    print(f" Training Baseline: TF-IDF + Logistic Regression")
    print(f"{'='*70}")
    
    classifier = BaselineTicketClassifier(confidence_threshold=args.confidence_threshold)
    classifier.model.max_iter = args.epochs
    classifier.model.random_state = args.seed
    
    classifier.load_data(args.train_data, args.val_data, args.test_data, args.label_mapping)
    classifier.train()
    val_metrics = classifier.evaluate(split='val')
    classifier.save(model_dir=args.output_dir)
    
    with open(os.path.join(args.output_dir, 'validation_metrics.json'), 'w') as f:
        json.dump(val_metrics, f, indent=2)
    
    return classifier, val_metrics


def train_distilbert(args):
    from src.models.train_distilbert import DistilBERTClassifier
    
    print(f"\n{'='*70}")
    print(f" Training DistilBERT")
    print(f"{'='*70}")
    
    classifier = DistilBERTClassifier(
        num_labels=9,
        confidence_threshold=args.confidence_threshold,
        max_length=args.max_length
    )
    
    classifier.load_data(args.train_data, args.val_data, args.test_data, args.label_mapping)
    classifier.train(
        output_dir=os.path.join(args.output_dir, 'checkpoints'),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience
    )
    val_metrics = classifier.evaluate(split='val')
    classifier.save(model_dir=args.output_dir)
    
    with open(os.path.join(args.output_dir, 'validation_metrics.json'), 'w') as f:
        json.dump(val_metrics, f, indent=2)
    
    return classifier, val_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Support Ticket Classification Model')
    
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'distilbert'])
    parser.add_argument('--train_data', type=str, default='data/processed/train.csv')
    parser.add_argument('--val_data', type=str, default='data/processed/val.csv')
    parser.add_argument('--test_data', type=str, default='data/processed/test.csv')
    parser.add_argument('--label_mapping', type=str, default='data/processed/label_mapping.json')
    
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--confidence_threshold', type=float, default=0.6)
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (defaults to models/baseline or models/distilbert)')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set default output directory based on model type
    if args.output_dir is None:
        if args.model == 'distilbert':
            args.output_dir = 'models/distilbert/final'
        else:
            args.output_dir = f'models/{args.model}'
    
    print(f"\n{'='*70}")
    print(f" SUPPORT TICKET CLASSIFICATION - TRAINING")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.model == 'baseline':
            classifier, val_metrics = train_baseline(args)
        elif args.model == 'distilbert':
            classifier, val_metrics = train_distilbert(args)
        else:
            raise ValueError(f"Unknown model: {args.model}")
        
        print(f"\n{'='*70}")
        print(f" TRAINING COMPLETE ✅")
        print(f"{'='*70}")
        print(f"Validation F1: {val_metrics['f1_weighted']:.4f}")
        print(f"Validation Acc: {val_metrics['accuracy']:.4f}")
        print(f"Saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
