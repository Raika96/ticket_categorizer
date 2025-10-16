"""
Example workflow for pushing and loading models from Hugging Face Hub
This script demonstrates the complete workflow for sharing your trained model.

Usage:
    1. Update REPO_NAME with your Hugging Face username
    2. Run this script to see the workflow
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def workflow_example():
    """
    Complete workflow example for Hugging Face integration
    """
    
    print("="*70)
    print("TicketCat - Hugging Face Workflow Example")
    print("="*70)
    
    # ========================================
    # STEP 1: Configuration
    # ========================================
    print("\n" + "="*70)
    print("STEP 1: Configuration")
    print("="*70)
    
    REPO_NAME = "your-username/ticketcat-distilbert"  # UPDATE THIS!
    LOCAL_MODEL_DIR = "models/distilbert/final"
    
    print(f"\nConfiguration:")
    print(f"  Repository name: {REPO_NAME}")
    print(f"  Local model dir: {LOCAL_MODEL_DIR}")
    
    # Check if model exists locally
    if not os.path.exists(LOCAL_MODEL_DIR):
        print(f"\n⚠️  Warning: Local model not found at {LOCAL_MODEL_DIR}")
        print("   Please train the model first using:")
        print("   python src/models/train_distilbert.py")
        return
    
    print(f"✅ Local model found")
    
    # ========================================
    # STEP 2: Push to Hugging Face (DEMO ONLY - NOT EXECUTED)
    # ========================================
    print("\n" + "="*70)
    print("STEP 2: Push to Hugging Face Hub")
    print("="*70)
    
    print("\nTo push your model to Hugging Face Hub, run:")
    print(f"\n  python src/models/push_to_huggingface.py \\")
    print(f"    --repo-name {REPO_NAME}")
    
    print("\nWhat this will do:")
    print("  1. Load the trained model from local directory")
    print("  2. Update model config with category labels")
    print("  3. Create a comprehensive model card (README.md)")
    print("  4. Upload everything to Hugging Face Hub")
    print("  5. Make your model accessible worldwide!")
    
    # Uncomment to actually push (requires authentication)
    # from src.models.push_to_huggingface import push_to_hub
    # push_to_hub(model_dir=LOCAL_MODEL_DIR, repo_name=REPO_NAME)
    
    # ========================================
    # STEP 3: Load from Hugging Face
    # ========================================
    print("\n" + "="*70)
    print("STEP 3: Load and Use Model from Hub")
    print("="*70)
    
    print("\nAfter pushing, you can load the model from anywhere:")
    print(f"\n  python src/models/load_from_huggingface.py \\")
    print(f"    --model-name {REPO_NAME} \\")
    print(f"    --demo")
    
    print("\nOr use in your code:")
    print(f"""
from src.models.load_from_huggingface import HuggingFaceTicketClassifier

# Load model
classifier = HuggingFaceTicketClassifier('{REPO_NAME}')

# Classify tickets
ticket = "I can't log into my account"
category, confidence = classifier.predict(ticket)
print(f"Category: {{category}} ({{confidence:.2f}})")
    """)
    
    # ========================================
    # STEP 4: Example Usage
    # ========================================
    print("\n" + "="*70)
    print("STEP 4: Example Usage Scenarios")
    print("="*70)
    
    example_scenarios = [
        {
            "title": "REST API Deployment",
            "code": """
from fastapi import FastAPI
from src.models.load_from_huggingface import HuggingFaceTicketClassifier

app = FastAPI()
classifier = HuggingFaceTicketClassifier('""" + REPO_NAME + """')

@app.post("/classify")
def classify_ticket(text: str):
    category, confidence = classifier.predict(text)
    return {
        "category": category,
        "confidence": confidence
    }
"""
        },
        {
            "title": "Batch Processing",
            "code": """
import pandas as pd
from src.models.load_from_huggingface import HuggingFaceTicketClassifier

# Load tickets
df = pd.read_csv('tickets.csv')
classifier = HuggingFaceTicketClassifier('""" + REPO_NAME + """')

# Classify all tickets
predictions, confidences = classifier.predict(df['text'].tolist())
df['category'] = predictions
df['confidence'] = confidences

# Save results
df.to_csv('tickets_classified.csv', index=False)
"""
        },
        {
            "title": "Real-time Classification",
            "code": """
from src.models.load_from_huggingface import HuggingFaceTicketClassifier

classifier = HuggingFaceTicketClassifier('""" + REPO_NAME + """')

# Classify incoming ticket
def handle_new_ticket(ticket_text):
    category, confidence = classifier.predict(ticket_text)
    
    if confidence < 0.6:
        # Route to human agent
        route_to_agent(ticket_text, category, confidence)
    else:
        # Auto-route to appropriate team
        route_to_team(category, ticket_text)
"""
        }
    ]
    
    for i, scenario in enumerate(example_scenarios, 1):
        print(f"\n{i}. {scenario['title']}:")
        print(scenario['code'])
    
    # ========================================
    # STEP 5: Benefits
    # ========================================
    print("\n" + "="*70)
    print("STEP 5: Benefits of Hugging Face Hub")
    print("="*70)
    
    benefits = [
        "🌍 Accessible from anywhere - no need to copy model files",
        "🔄 Version control - track model improvements over time",
        "👥 Easy sharing - share with team or community",
        "📊 Model cards - automatic documentation",
        "🚀 Simple deployment - works with any Transformers-compatible platform",
        "💾 Free hosting - no infrastructure costs",
        "🔍 Discoverability - others can find and use your model",
        "🔒 Private repos - keep proprietary models secure"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("Quick Start Commands")
    print("="*70)
    
    print("""
1. Login to Hugging Face:
   huggingface-cli login

2. Push your model:
   python src/models/push_to_huggingface.py --repo-name """ + REPO_NAME + """

3. Test loading:
   python src/models/load_from_huggingface.py --model-name """ + REPO_NAME + """ --demo

4. Use in production:
   See examples above or check HUGGINGFACE_GUIDE.md
""")
    
    print("="*70)
    print("For more details, see: HUGGINGFACE_GUIDE.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    workflow_example()

