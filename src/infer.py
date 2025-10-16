"""
Inference utilities for ticket classification
Provides model loading and prediction functions for the API
"""

import os
import sys
import joblib
import json
from typing import Dict, List, Tuple
from datetime import datetime
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessing.text_cleaner import TextCleaner


def load_baseline_model(model_dir='models/baseline'):
    """Load baseline TF-IDF + LogisticRegression model"""
    model_path = os.path.join(project_root, model_dir)
    
    print(f"Loading baseline model from {model_path}...")
    
    # Load components
    vectorizer = joblib.load(os.path.join(model_path, 'vectorizer.joblib'))
    model = joblib.load(os.path.join(model_path, 'model.joblib'))
    
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config = json.load(f)
    
    text_cleaner = TextCleaner()
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'config': config,
        'text_cleaner': text_cleaner,
        'type': 'baseline'
    }


def load_distilbert_model(model_dir='models/distilbert/final'):
    """Load DistilBERT model"""
    model_path = os.path.join(project_root, model_dir)
    
    print(f"Loading DistilBERT model from {model_path}...")
    
    # Set device
    device = torch.device('cpu')  # Use CPU to avoid MPS issues
    
    # Load model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load config
    with open(os.path.join(model_path, 'classifier_config.json'), 'r') as f:
        config = json.load(f)
    
    text_cleaner = TextCleaner()
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'config': config,
        'text_cleaner': text_cleaner,
        'device': device,
        'type': 'distilbert'
    }


def predict_ticket(ticket_data: Dict, model_dict: Dict) -> Dict:
    """Predict category for a single ticket"""
    # Combine title and description
    text = f"{ticket_data['title']} {ticket_data['description']}"
    
    # Clean text
    text_cleaner = model_dict['text_cleaner']
    cleaned_text = text_cleaner.clean_text(text)
    
    config = model_dict['config']
    model_type = model_dict['type']
    confidence_threshold = config.get('confidence_threshold', 0.6)
    
    if model_type == 'baseline':
        # Baseline prediction
        model = model_dict['model']
        vectorizer = model_dict['vectorizer']
        
        # Vectorize
        X = vectorizer.transform([cleaned_text])
        
        # Predict
        probas = model.predict_proba(X)[0]
        pred_idx = probas.argmax()
        confidence = float(probas[pred_idx])
        
        # Get category
        id_to_category = config['label_mapping']['id_to_category']
        predicted_category = id_to_category[str(pred_idx)]
        
        # All probabilities
        all_probas = {
            id_to_category[str(i)]: float(probas[i])
            for i in range(len(probas))
        }
        
    else:  # distilbert
        # DistilBERT prediction
        model = model_dict['model']
        tokenizer = model_dict['tokenizer']
        device = model_dict['device']
        
        # Tokenize
        inputs = tokenizer(
            cleaned_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probas = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        pred_idx = probas.argmax()
        confidence = float(probas[pred_idx])
        
        # Get category
        id_to_category = config['label_mapping']['id_to_category']
        predicted_category = id_to_category[str(pred_idx)]
        
        # All probabilities
        all_probas = {
            id_to_category[str(i)]: float(probas[i])
            for i in range(len(probas))
        }
    
    # Determine if manual review needed
    needs_manual_review = confidence < confidence_threshold
    
    return {
        'predicted_category': predicted_category,
        'confidence': confidence,
        'needs_manual_review': needs_manual_review,
        'model_type': model_type,
        'timestamp': datetime.now().isoformat(),
        'all_probabilities': all_probas
    }


def predict_batch(tickets: List[Dict], model_dict: Dict) -> List[Dict]:
    """Predict categories for a batch of tickets"""
    return [predict_ticket(ticket, model_dict) for ticket in tickets]


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CLI for ticket classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # From command line argument
  python3 src/infer.py '{"title": "Cannot login", "description": "Password not working"}'
  
  # From stdin
  echo '{"title": "Refund request", "description": "Want my money back"}' | python3 src/infer.py
  
  # Use baseline model
  python3 src/infer.py --model baseline '{"title": "Bug report", "description": "App crashes"}'
  
  # Custom model directory
  python3 src/infer.py --model-dir models/distilbert/checkpoints/checkpoint-6195 '{"title": "Test", "description": "Test"}'
        '''
    )
    
    parser.add_argument(
        'json_input',
        nargs='?',
        help='JSON input: {"title": "...", "description": "..."} (reads from stdin if not provided)'
    )
    parser.add_argument(
        '--model',
        choices=['baseline', 'distilbert'],
        default='distilbert',
        help='Model to use for prediction (default: distilbert)'
    )
    parser.add_argument(
        '--model-dir',
        help='Custom model directory (optional)'
    )
    parser.add_argument(
        '--compact',
        action='store_true',
        help='Output compact JSON (minimal fields)'
    )
    
    args = parser.parse_args()
    
    # Get JSON input
    if args.json_input:
        json_str = args.json_input
    else:
        # Read from stdin
        json_str = sys.stdin.read().strip()
    
    if not json_str:
        parser.error('No JSON input provided')
    
    # Parse JSON
    try:
        ticket_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate input
    if 'title' not in ticket_data or 'description' not in ticket_data:
        print('Error: JSON must contain "title" and "description" fields', file=sys.stderr)
        sys.exit(1)
    
    # Load model
    try:
        if args.model == 'baseline':
            model_dir = args.model_dir or 'models/baseline'
            model_dict = load_baseline_model(model_dir)
        else:
            model_dir = args.model_dir or 'models/distilbert/final'
            model_dict = load_distilbert_model(model_dir)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Make prediction
    try:
        result = predict_ticket(ticket_data, model_dict)
        
        # Format output
        if args.compact:
            # Minimal output (matches requirement)
            output = {
                'predicted_category': result['predicted_category'],
                'confidence': result['confidence']
            }
        else:
            # Full output
            output = result
        
        # Print JSON result
        print(json.dumps(output, indent=2))
        
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)
