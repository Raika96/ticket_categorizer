# Hugging Face Hub Integration Guide

This guide explains how to push your trained DistilBERT model to Hugging Face Hub and load it for inference.

## Prerequisites

1. **Hugging Face Account**: Create an account at [huggingface.co](https://huggingface.co)
2. **API Token**: Get your token from [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. **Required Packages**: Install dependencies if needed
   ```bash
   pip install huggingface_hub transformers torch
   ```

## Quick Start

### 1. Login to Hugging Face

First, login using the CLI (recommended):
```bash
huggingface-cli login
```

Or set your token as an environment variable:
```bash
export HF_TOKEN="your_token_here"
```

### 2. Push Model to Hub

Run the push script:
```bash
python src/models/push_to_huggingface.py --repo-name your-username/ticketcat-distilbert
```

**Options:**
- `--model-dir`: Path to local model directory (default: `models/distilbert/final`)
- `--repo-name`: Repository name on Hub (format: `username/model-name`)
- `--private`: Make the repository private
- `--token`: Provide token directly (not recommended, use login instead)

**Example:**
```bash
# Public repository
python src/models/push_to_huggingface.py --repo-name myusername/ticketcat-distilbert

# Private repository
python src/models/push_to_huggingface.py --repo-name myusername/ticketcat-distilbert --private
```

### 3. Load and Use Model

Once uploaded, you can load the model from anywhere:

```bash
# Run demo with example tickets
python src/models/load_from_huggingface.py --model-name your-username/ticketcat-distilbert --demo

# Classify a single ticket
python src/models/load_from_huggingface.py \
  --model-name your-username/ticketcat-distilbert \
  --text "I can't log into my account"
```

## Python Usage Examples

### Basic Usage

```python
from src.models.load_from_huggingface import HuggingFaceTicketClassifier

# Load model
classifier = HuggingFaceTicketClassifier(
    model_name='your-username/ticketcat-distilbert',
    confidence_threshold=0.6
)

# Classify a single ticket
ticket = "The application is running very slowly"
category, confidence = classifier.predict(ticket)
print(f"Category: {category} (confidence: {confidence:.2f})")

# Classify multiple tickets
tickets = [
    "Can't reset my password",
    "Integration with API not working",
    "Charged twice this month"
]
predictions, confidences = classifier.predict(tickets)
for ticket, cat, conf in zip(tickets, predictions, confidences):
    print(f"{ticket[:30]:30} -> {cat:30} ({conf:.2f})")
```

### Get Top-K Predictions

```python
# Get top 3 predictions with probabilities
ticket = "System is down and not responding"
top_predictions = classifier.predict_with_probabilities(ticket, top_k=3)

print(f"Ticket: {ticket}")
for i, (category, probability) in enumerate(top_predictions, 1):
    print(f"  {i}. {category}: {probability:.4f}")
```

### Using Directly with Transformers

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "your-username/ticketcat-distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Classify
ticket = "Need help integrating with Salesforce"
inputs = tokenizer(ticket, return_tensors="pt", padding=True, truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs.max().item()

# Get category name
category = model.config.id2label[predicted_class]
print(f"Category: {category}, Confidence: {confidence:.4f}")
```

## Model Categories

The model classifies tickets into 9 categories:

1. **Account Access / Login Issues** - Password resets, login problems, account lockouts
2. **Billing & Payments** - Charges, refunds, subscription issues
3. **Bug / Defect Reports** - Software bugs, errors, crashes
4. **Feature Requests** - New feature suggestions, enhancements
5. **General Inquiries / Other** - General questions, uncategorized
6. **How-To / Product Usage Questions** - Usage help, documentation requests
7. **Integration Issues** - API integration, third-party integrations
8. **Performance Problems** - Slow performance, timeouts, lag
9. **Security & Compliance** - Security concerns, compliance questions

## Confidence Threshold

The default confidence threshold is **0.6**. Predictions with confidence below this threshold are marked as "Uncategorized" and may need human review.

You can adjust this when initializing the classifier:

```python
# More conservative (fewer uncategorized, but potentially less accurate)
classifier = HuggingFaceTicketClassifier(
    model_name='your-username/ticketcat-distilbert',
    confidence_threshold=0.5
)

# More strict (more uncategorized, but higher accuracy on classified)
classifier = HuggingFaceTicketClassifier(
    model_name='your-username/ticketcat-distilbert',
    confidence_threshold=0.8
)
```

## Integration with Your Application

### REST API Integration

The model works seamlessly with the existing API (`src/api.py`). After uploading to Hugging Face, you can modify the inference code to load from the hub instead of local files.

### Batch Processing

For batch processing of tickets:

```python
import pandas as pd
from src.models.load_from_huggingface import HuggingFaceTicketClassifier

# Load classifier
classifier = HuggingFaceTicketClassifier('your-username/ticketcat-distilbert')

# Load tickets
df = pd.read_csv('tickets.csv')

# Classify all tickets
predictions, confidences = classifier.predict(df['ticket_text'].tolist())

# Add to dataframe
df['predicted_category'] = predictions
df['confidence'] = confidences

# Save results
df.to_csv('tickets_classified.csv', index=False)

# Filter low confidence predictions for review
low_confidence = df[df['confidence'] < 0.6]
print(f"Tickets needing review: {len(low_confidence)}")
```

## Updating the Model

To update the model after retraining:

1. Train new model locally
2. Run the push script with the `--repo-name` pointing to the same repository
3. The new version will be uploaded and versioned automatically

```bash
python src/models/push_to_huggingface.py --repo-name your-username/ticketcat-distilbert
```

## Troubleshooting

### Authentication Issues

If you get authentication errors:
```bash
# Re-login
huggingface-cli logout
huggingface-cli login
```

### Model Not Found

Make sure you're using the correct format: `username/model-name`

```python
# ✅ Correct
classifier = HuggingFaceTicketClassifier('myusername/ticketcat-distilbert')

# ❌ Incorrect
classifier = HuggingFaceTicketClassifier('ticketcat-distilbert')
```

### Out of Memory

If you run out of memory during inference:
- Use CPU instead of GPU (model automatically detects)
- Process tickets in smaller batches
- Reduce `max_length` parameter

```python
# Process in batches of 32
batch_size = 32
all_predictions = []

for i in range(0, len(tickets), batch_size):
    batch = tickets[i:i+batch_size]
    preds, _ = classifier.predict(batch)
    all_predictions.extend(preds)
```

## Model Card

When you push the model, an automatic model card (README.md) is generated with:
- Model description
- Categories
- Performance metrics
- Usage examples
- Training details

You can view and edit this on the Hugging Face Hub after uploading.

## Resources

- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index)
- [Transformers Library](https://huggingface.co/docs/transformers/index)
- [Model Sharing Guide](https://huggingface.co/docs/transformers/model_sharing)

## Next Steps

1. Push your model to Hugging Face Hub
2. Test loading it from the hub
3. Update your production code to use the hub model
4. Share your model with your team or community
5. Monitor model performance and retrain as needed

