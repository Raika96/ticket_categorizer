"""
Script to load the DistilBERT model from Hugging Face Hub

CONFIGURE YOUR MODEL:
Replace 'YOUR_HF_USERNAME' with your Hugging Face username in the DEFAULT_MODEL_NAME below
Example: DEFAULT_MODEL_NAME = 'johndoe/ticketcat-distilbert'
"""

import torch
import argparse
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

DEFAULT_MODEL_NAME = 'raika96/ticket_cat_bert'


class HuggingFaceTicketClassifier:
    """
    Wrapper class for loading and using the TicketCat model from Hugging Face Hub
    """
    
    def __init__(self, model_name=None, confidence_threshold=0.6):
        """
        Initialize the classifier with a model from Hugging Face Hub
        
        Args:
            model_name: Name of the model on Hugging Face Hub (username/model-name)
                       If None, uses DEFAULT_MODEL_NAME configured at top of file
            confidence_threshold: Minimum confidence for predictions (0-1)
        """
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model from Hugging Face Hub: {self.model_name}")
        print(f"Device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get label mappings from model config
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        self.num_labels = len(self.id2label)
        
        print(f"Model loaded successfully")
        print(f"   Number of categories: {self.num_labels}")
        print(f"   Confidence threshold: {self.confidence_threshold}")
    
    def predict(self, texts, return_confidence=True, max_length=128):
        """
        Predict categories for one or more ticket texts
        
        Args:
            texts: Single text string or list of text strings
            return_confidence: Whether to return confidence scores
            max_length: Maximum sequence length for tokenization
            
        Returns:
            If return_confidence=True: (predictions, confidences)
            If return_confidence=False: predictions
            
            Where predictions is a list of category names
            and confidences is a list of confidence scores (0-1)
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                pred_id = torch.argmax(probs, dim=1).item()
                confidence = probs.max().item()
                
                # Map to category name
                pred_label = self.id2label[pred_id]
                
                # Handle low confidence predictions
                if confidence < self.confidence_threshold:
                    pred_label = 'Uncategorized'
                
                predictions.append(pred_label)
                confidences.append(confidence)
        
        # Return single values if single input
        if single_input:
            predictions = predictions[0]
            confidences = confidences[0]
        
        if return_confidence:
            return predictions, confidences
        else:
            return predictions
    
    def predict_with_probabilities(self, text, max_length=128, top_k=3):
        """
        Get top-k predictions with probabilities for a single text
        
        Args:
            text: Input text string
            max_length: Maximum sequence length
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples (category_name, probability)
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_labels))
            
            results = [
                (self.id2label[idx.item()], prob.item())
                for idx, prob in zip(top_indices, top_probs)
            ]
            
            return results
    
    def get_categories(self):
        """Get list of all possible categories"""
        return list(self.label2id.keys())
    
    def get_model_info(self):
        """Get information about the model"""
        return {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'categories': self.get_categories(),
            'confidence_threshold': self.confidence_threshold,
            'device': str(self.device)
        }
    
    def save_to_local(self, local_dir='models/distilbert/final'):
        """
        Save the model from Hugging Face Hub to local directory
        This will replace existing local model files
        
        Args:
            local_dir: Local directory to save the model to
        """
        import os
        import json
        from datetime import datetime
        
        print(f"\nSaving model to local directory: {local_dir}")
        
        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(local_dir)
        self.tokenizer.save_pretrained(local_dir)
        print(f"   Model and tokenizer saved")
        
        # Create classifier config (compatible with train_distilbert.py)
        classifier_config = {
            'model_type': 'distilbert_sequence_classification',
            'confidence_threshold': self.confidence_threshold,
            'max_length': 128,
            'num_labels': self.num_labels,
            'training_stats': {},
            'label_mapping': {
                'id_to_category': self.id2label,
                'label_to_id': self.label2id,
                'num_classes': self.num_labels
            },
            'created_at': datetime.now().isoformat(),
            'source': f'Downloaded from Hugging Face Hub: {self.model_name}'
        }
        
        config_path = os.path.join(local_dir, 'classifier_config.json')
        with open(config_path, 'w') as f:
            json.dump(classifier_config, f, indent=2)
        print(f"   Classifier config saved")
        
        print(f"\nModel successfully saved to {local_dir}")
        print(f"   You can now load it locally using:")
        print(f"   from src.models.train_distilbert import DistilBERTClassifier")
        print(f"   classifier = DistilBERTClassifier.load('{local_dir}')")
        
        return local_dir


def demo_usage(model_name=None):
    """Demo function showing how to use the classifier"""
    
    print("\n" + "="*60)
    print("TicketCat Demo - Hugging Face Model Loader")
    print("="*60)
    
    # Load model
    classifier = HuggingFaceTicketClassifier(model_name=model_name or DEFAULT_MODEL_NAME)
    
    # Example tickets
    example_tickets = [
        "I can't log into my account, password reset is not working",
        "The application is running very slowly and timing out",
        "How do I integrate your API with Salesforce?",
        "I was charged twice for my subscription this month",
        "The search feature is not returning any results",
    ]
    
    print("\n" + "-"*60)
    print("Example Predictions:")
    print("-"*60)
    
    for i, ticket in enumerate(example_tickets, 1):
        category, confidence = classifier.predict(ticket)
        
        print(f"\n{i}. Ticket: {ticket[:60]}...")
        print(f"   Category: {category}")
        print(f"   Confidence: {confidence:.4f}")
        
        # Show top-3 predictions
        top_preds = classifier.predict_with_probabilities(ticket, top_k=3)
        print(f"   Top 3 predictions:")
        for j, (cat, prob) in enumerate(top_preds, 1):
            print(f"      {j}. {cat}: {prob:.4f}")
    
    print("\n" + "-"*60)
    print("Model Information:")
    print("-"*60)
    info = classifier.get_model_info()
    for key, value in info.items():
        if key != 'categories':
            print(f"   {key}: {value}")
    
    print(f"\n   Available categories:")
    for cat in info['categories']:
        print(f"      - {cat}")
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and use TicketCat model from Hugging Face Hub')
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help=f'Model name on Hugging Face Hub (default: {DEFAULT_MODEL_NAME})'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo with example tickets'
    )
    parser.add_argument(
        '--text',
        type=str,
        default=None,
        help='Single ticket text to classify'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.6,
        help='Minimum confidence threshold (0-1)'
    )
    parser.add_argument(
        '--save-local',
        type=str,
        default=None,
        metavar='DIR',
        help='Download and save model to local directory (e.g., models/distilbert/final)'
    )
    
    args = parser.parse_args()
    
    if args.save_local:
        # Download from HF and save to local directory
        print(f"\nDownloading model from Hugging Face Hub and saving locally...")
        classifier = HuggingFaceTicketClassifier(
            model_name=args.model_name or DEFAULT_MODEL_NAME,
            confidence_threshold=args.confidence_threshold
        )
        classifier.save_to_local(local_dir=args.save_local)
        print(f"\nDone! Model saved to {args.save_local}")
    elif args.demo:
        # Run demo
        demo_usage(model_name=args.model_name or DEFAULT_MODEL_NAME)
    elif args.text:
        # Classify single text
        classifier = HuggingFaceTicketClassifier(
            model_name=args.model_name or DEFAULT_MODEL_NAME,
            confidence_threshold=args.confidence_threshold
        )
        category, confidence = classifier.predict(args.text)
        
        print(f"\nTicket: {args.text}")
        print(f"Category: {category}")
        print(f"Confidence: {confidence:.4f}\n")
        
        # Show top predictions
        print("Top 3 predictions:")
        top_preds = classifier.predict_with_probabilities(args.text, top_k=3)
        for i, (cat, prob) in enumerate(top_preds, 1):
            print(f"  {i}. {cat}: {prob:.4f}")
    else:
        # Just show usage
        print("TicketCat - Hugging Face Model Loader")
        print("\nUsage:")
        print("  python load_from_huggingface.py --demo                    # Run demo")
        print("  python load_from_huggingface.py --text 'your ticket'     # Classify text")
        print("  python load_from_huggingface.py --model-name user/model  # Use specific model")
        print("\nExample:")
        print("  python load_from_huggingface.py --text 'Cannot login to account'")

