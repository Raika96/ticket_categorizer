# Hugging Face Integration - Quick Start

A 5-minute guide to sharing your TicketCat model on Hugging Face Hub.

## Prerequisites

✅ Trained DistilBERT model in `models/distilbert/final/`  
✅ Hugging Face account ([sign up here](https://huggingface.co/join))  
✅ `huggingface_hub` package installed

```bash
pip install huggingface_hub
```

## Step-by-Step Guide

### 1. Login to Hugging Face (One-Time)

```bash
huggingface-cli login
```

Enter your API token when prompted. Get your token from: https://huggingface.co/settings/tokens

### 2. Push Your Model

```bash
python src/models/push_to_huggingface.py --repo-name YOUR_USERNAME/ticketcat-distilbert
```

**Replace `YOUR_USERNAME` with your Hugging Face username!**

This will:
- ✅ Upload your trained model
- ✅ Create a model card with documentation
- ✅ Make it accessible via simple model name
- ✅ Add category labels to the config

**Expected output:**
```
🚀 Pushing model to Hugging Face Hub...
   Model directory: models/distilbert/final
   Repository name: YOUR_USERNAME/ticketcat-distilbert

📦 Loading model and tokenizer...
   ✅ Model and tokenizer loaded successfully

📝 Creating model card...
   ✅ Model card saved to models/distilbert/final/README.md

🏗️  Creating repository: YOUR_USERNAME/ticketcat-distilbert
   ✅ Repository created/verified

⬆️  Uploading model to Hugging Face Hub...
   ✅ Model uploaded successfully
   ✅ Tokenizer uploaded successfully
   ✅ Model card uploaded successfully

✨ Success! Model available at: https://huggingface.co/YOUR_USERNAME/ticketcat-distilbert
```

### 3. Test Your Model

Run the demo to verify it works:

```bash
python src/models/load_from_huggingface.py \
  --model-name YOUR_USERNAME/ticketcat-distilbert \
  --demo
```

You should see example ticket classifications!

### 4. Use in Your Code

```python
from src.models.load_from_huggingface import HuggingFaceTicketClassifier

# Load your model from anywhere!
classifier = HuggingFaceTicketClassifier('YOUR_USERNAME/ticketcat-distilbert')

# Classify a ticket
category, confidence = classifier.predict("Can't reset my password")
print(f"{category} ({confidence:.2%})")
```

## Common Options

### Make Repository Private

```bash
python src/models/push_to_huggingface.py \
  --repo-name YOUR_USERNAME/ticketcat-distilbert \
  --private
```

### Use Custom Model Directory

```bash
python src/models/push_to_huggingface.py \
  --model-dir path/to/your/model \
  --repo-name YOUR_USERNAME/ticketcat-distilbert
```

### Adjust Confidence Threshold

```python
classifier = HuggingFaceTicketClassifier(
    'YOUR_USERNAME/ticketcat-distilbert',
    confidence_threshold=0.7  # More strict
)
```

## What Gets Uploaded?

📦 **Model Files:**
- `model.safetensors` - Model weights (250MB)
- `config.json` - Model configuration
- `classifier_config.json` - TicketCat-specific config

🔤 **Tokenizer Files:**
- `vocab.txt` - Vocabulary
- `tokenizer_config.json` - Tokenizer settings
- `special_tokens_map.json` - Special tokens

📄 **Documentation:**
- `README.md` - Auto-generated model card
- Usage examples
- Performance metrics
- Category descriptions

## Example Use Cases

### 1. REST API

```python
from fastapi import FastAPI
from src.models.load_from_huggingface import HuggingFaceTicketClassifier

app = FastAPI()
classifier = HuggingFaceTicketClassifier('YOUR_USERNAME/ticketcat-distilbert')

@app.post("/classify")
def classify(text: str):
    category, confidence = classifier.predict(text)
    return {"category": category, "confidence": confidence}
```

### 2. Batch Processing

```python
import pandas as pd
from src.models.load_from_huggingface import HuggingFaceTicketClassifier

df = pd.read_csv('tickets.csv')
classifier = HuggingFaceTicketClassifier('YOUR_USERNAME/ticketcat-distilbert')

# Classify all tickets at once
predictions, confidences = classifier.predict(df['text'].tolist())
df['category'] = predictions
df['confidence'] = confidences

df.to_csv('classified_tickets.csv')
```

### 3. Direct Transformers Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('YOUR_USERNAME/ticketcat-distilbert')
tokenizer = AutoTokenizer.from_pretrained('YOUR_USERNAME/ticketcat-distilbert')

# Model is ready to use!
```

## Troubleshooting

### "Authentication required"

Run `huggingface-cli login` first.

### "Repository not found"

Make sure you're using the format: `username/model-name` (not just `model-name`)

### "Model files not found"

Train the model first:
```bash
python src/models/train_distilbert.py
```

### Out of memory when loading

The model is ~250MB. Make sure you have enough RAM. Use CPU if GPU memory is limited:
```python
# Model automatically uses CPU if GPU unavailable
classifier = HuggingFaceTicketClassifier('YOUR_USERNAME/ticketcat-distilbert')
```

## Next Steps

- ✨ Share your model with your team
- 📊 Track model versions on Hugging Face
- 🚀 Deploy to production using the hub
- 🔄 Update model by re-running push script
- 📈 Monitor usage and performance

## Resources

- 📚 [Full Guide](HUGGINGFACE_GUIDE.md) - Comprehensive documentation
- 🧪 [Example Script](examples/huggingface_workflow_example.py) - Full workflow demo
- 📖 [Models README](src/models/README.md) - Model documentation
- 🔗 [Hugging Face Docs](https://huggingface.co/docs/hub/index) - Official documentation

---

**Questions?** Check [HUGGINGFACE_GUIDE.md](HUGGINGFACE_GUIDE.md) for detailed explanations and advanced usage.

