"""
GPT Fallback for uncertain predictions
Uses GPT to classify tickets when ML model confidence is low
"""

import os
import sys
from typing import Dict, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from load_secrets import load_secrets


# Categories
CATEGORIES = [
    "Account Access / Login Issues",
    "Billing & Payments",
    "Bug / Defect Reports",
    "Feature Requests",
    "General Inquiries / Other",
    "How-To / Product Usage Questions",
    "Integration Issues",
    "Performance Problems",
    "Security & Compliance"
]


def classify_with_gpt(title: str, description: str, threshold: float = 0.7) -> Optional[Dict]:
    """
    Use GPT to classify a ticket
    
    Returns:
        Dict with prediction or None if GPT can't classify with confidence
    """
    try:
        import openai
        
        # Get API key
        secrets = load_secrets()
        api_key = secrets['openai']['api_key']
        if not api_key:
            return None
        
        openai.api_key = api_key
        
        # Create prompt
        prompt = f"""You are a support ticket classifier. Classify the following support ticket into ONE of these categories:

{chr(10).join(f'- {cat}' for cat in CATEGORIES)}

Ticket Title: {title}
Ticket Description: {description}

Respond with ONLY the category name, or "UNCERTAIN" if you cannot confidently classify it.
Category:"""
        
        # Call GPT
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a support ticket classification assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=50
        )
        
        predicted_category = response.choices[0].message.content.strip()
        
        # Check if valid category
        if predicted_category in CATEGORIES:
            return {
                'predicted_category': predicted_category,
                'confidence': threshold,  # GPT doesn't provide confidence, use threshold
                'method': 'gpt'
            }
        elif predicted_category.upper() == "UNCERTAIN":
            return {
                'predicted_category': 'Uncategorized',
                'confidence': 0.0,
                'method': 'gpt_unknown'
            }
        else:
            # GPT returned something unexpected
            return None
            
    except Exception as e:
        print(f"GPT fallback error: {e}")
        return None


def hybrid_classify(ticket_data: Dict, ml_prediction: Dict, use_gpt_fallback: bool = True) -> Dict:
    """
    Hybrid classification: Use ML first, fallback to GPT if confidence is low
    
    Args:
        ticket_data: Dict with 'title' and 'description'
        ml_prediction: ML model prediction dict
        use_gpt_fallback: Whether to use GPT fallback for low confidence
        
    Returns:
        Final prediction dict
    """
    confidence_threshold = 0.6
    
    # If ML is confident, use it
    if ml_prediction['confidence'] >= confidence_threshold:
        return {
            **ml_prediction,
            'classification_method': 'ml'
        }
    
    # If GPT fallback is disabled, return ML prediction anyway
    if not use_gpt_fallback:
        return {
            **ml_prediction,
            'classification_method': 'ml'
        }
    
    # Try GPT fallback
    gpt_result = classify_with_gpt(
        ticket_data['title'],
        ticket_data['description']
    )
    
    if gpt_result and gpt_result['predicted_category'] != 'Uncategorized':
        # GPT classified successfully
        return {
            'predicted_category': gpt_result['predicted_category'],
            'confidence': gpt_result['confidence'],
            'needs_manual_review': False,
            'model_type': ml_prediction['model_type'],
            'classification_method': 'gpt',
            'ml_prediction': ml_prediction,
            'gpt_prediction': gpt_result
        }
    else:
        # GPT also uncertain or failed, mark for manual review
        return {
            **ml_prediction,
            'needs_manual_review': True,
            'classification_method': 'gpt_unknown' if gpt_result else 'ml',
            'ml_prediction': ml_prediction,
            'gpt_prediction': gpt_result
        }

