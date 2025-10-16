"""
Script to push the trained DistilBERT model to Hugging Face Hub
"""

import os
import json
import argparse
from huggingface_hub import HfApi, create_repo
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


def create_model_card(label_mapping, metrics_path=None):
    """Create a README model card for the model"""
    
    # Load metrics if available
    metrics_info = ""
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            metrics_info = f"""
## Performance Metrics

- **Accuracy**: {metrics.get('accuracy', 'N/A')}
- **F1 Score (Weighted)**: {metrics.get('f1_weighted', 'N/A')}
- **Precision (Weighted)**: {metrics.get('precision_weighted', 'N/A')}
- **Recall (Weighted)**: {metrics.get('recall_weighted', 'N/A')}
"""
    
    categories_list = "\n".join([f"- {cat}" for cat in label_mapping['label_to_id'].keys()])
    
    model_card = f"""---
language: en
license: apache-2.0
tags:
- text-classification
- distilbert
- ticket-classification
- customer-support
- it-support
datasets:
- custom
metrics:
- accuracy
- f1
widget:
- text: "I can't log into my account, password reset not working"
- text: "System is very slow, taking forever to load pages"
- text: "How do I integrate with Salesforce API?"
---

# TicketCat: IT Support Ticket Classification Model

This is a fine-tuned DistilBERT model for classifying IT support tickets into 9 categories.

## Model Description

This model was fine-tuned on customer support tickets to automatically categorize incoming IT support requests. It uses DistilBERT as the base model and was trained to classify tickets into 9 distinct categories.

## Intended Use

This model is designed to automatically categorize IT support tickets to help route them to the appropriate support team or department.

## Categories

The model classifies tickets into the following categories:

{categories_list}
{metrics_info}
## Usage

```python
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Load model and tokenizer
model_name = "YOUR_HF_USERNAME/ticketcat-distilbert"  # Replace with your actual model name
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Classify a ticket
ticket_text = "I can't access my account, password reset link is not working"
inputs = tokenizer(ticket_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs.max().item()

# Map to category name
categories = {label_mapping['id_to_category']}
predicted_category = categories[str(predicted_class)]

print(f"Category: {{predicted_category}}")
print(f"Confidence: {{confidence:.4f}}")
```

## Training Details

- **Base Model**: distilbert-base-uncased
- **Framework**: Hugging Face Transformers
- **Task**: Multi-class text classification
- **Number of Classes**: 9
- **Max Sequence Length**: 128 tokens
- **Training Approach**: Fine-tuning with class weights for imbalanced data

## Limitations

- The model was trained on IT/customer support tickets and may not perform well on other domains
- Performance may vary on tickets that don't fit clearly into one category
- Low confidence predictions (< 0.6) may need human review

## Citation

If you use this model, please cite:

```
@misc{{ticketcat2024,
  author = {{TicketCat Team}},
  title = {{TicketCat: IT Support Ticket Classification}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/YOUR_USERNAME/ticketcat-distilbert}}}}
}}
```
"""
    return model_card


def push_to_hub(
    model_dir='models/distilbert/final',
    repo_name='ticketcat-distilbert',
    private=False,
    token=None
):
    """
    Push the trained model to Hugging Face Hub
    
    Args:
        model_dir: Path to the local model directory
        repo_name: Name for the repository on Hugging Face Hub
        private: Whether to make the repository private
        token: Hugging Face API token (if not set in environment)
    """
    
    print(f"Pushing model to Hugging Face Hub...")
    print(f"   Model directory: {model_dir}")
    print(f"   Repository name: {repo_name}")
    
    # Load the config to get label mappings
    config_path = os.path.join(model_dir, 'classifier_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        classifier_config = json.load(f)
    
    label_mapping = classifier_config.get('label_mapping', {})
    
    # Load model and tokenizer to verify they exist
    print("\nLoading model and tokenizer...")
    try:
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        print("   Model and tokenizer loaded successfully")
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")
    
    # Update model config with proper label mappings
    model.config.id2label = {int(k): v for k, v in label_mapping['id_to_category'].items()}
    model.config.label2id = label_mapping['label_to_id']
    
    # Create model card
    print("\nCreating model card...")
    model_card = create_model_card(label_mapping)
    readme_path = os.path.join(model_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(model_card)
    print(f"   Model card saved to {readme_path}")
    
    # Create repository
    print(f"\nCreating repository: {repo_name}")
    api = HfApi(token=token)
    
    try:
        repo_url = create_repo(
            repo_id=repo_name,
            private=private,
            token=token,
            exist_ok=True
        )
        print(f"   Repository created/verified: {repo_url}")
    except Exception as e:
        print(f"   Warning: Repository may already exist: {str(e)}")
    
    # Push model to hub
    print(f"\nUploading model to Hugging Face Hub...")
    try:
        model.push_to_hub(
            repo_id=repo_name,
            token=token,
            private=private
        )
        print("   Model uploaded successfully")
        
        tokenizer.push_to_hub(
            repo_id=repo_name,
            token=token,
            private=private
        )
        print("   Tokenizer uploaded successfully")
        
        # Upload README
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_name,
            token=token
        )
        print("   Model card uploaded successfully")
        
        # Upload classifier config for reference
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="classifier_config.json",
            repo_id=repo_name,
            token=token
        )
        print("   Classifier config uploaded successfully")
        
    except Exception as e:
        raise Exception(f"Failed to upload to hub: {str(e)}")
    
    print(f"\nSuccess! Model available at: https://huggingface.co/{repo_name}")
    print(f"\nTo use this model:")
    print(f"   from transformers import AutoModelForSequenceClassification, AutoTokenizer")
    print(f"   model = AutoModelForSequenceClassification.from_pretrained('{repo_name}')")
    print(f"   tokenizer = AutoTokenizer.from_pretrained('{repo_name}')")
    
    return repo_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Push trained model to Hugging Face Hub')
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/distilbert/final',
        help='Path to the local model directory'
    )
    parser.add_argument(
        '--repo-name',
        type=str,
        default='ticketcat-distilbert',
        help='Name for the repository on Hugging Face Hub'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make the repository private'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='Hugging Face API token (or set HF_TOKEN environment variable)'
    )
    
    args = parser.parse_args()
    
    # Check for token
    token = args.token or os.environ.get('HF_TOKEN')
    if not token:
        print("Warning: No Hugging Face token provided.")
        print("   Please either:")
        print("   1. Set HF_TOKEN environment variable")
        print("   2. Pass --token argument")
        print("   3. Login using: huggingface-cli login")
        print("\nAttempting to use cached credentials...")
    
    try:
        push_to_hub(
            model_dir=args.model_dir,
            repo_name=args.repo_name,
            private=args.private,
            token=token
        )
    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)

