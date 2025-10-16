# Hugging Face Setup Instructions

## For Model Owners (You)

### 1. Update the Configuration

Edit `src/models/load_from_huggingface.py` and replace `YOUR_HF_USERNAME` with your actual Hugging Face username:

```python
# Line 16 in src/models/load_from_huggingface.py
DEFAULT_MODEL_NAME = 'your-actual-username/ticketcat-distilbert'
```

### 2. Push Your Model to Hugging Face

```bash
# Login to Hugging Face (one-time)
huggingface-cli login

# Push your trained model
python src/models/push_to_huggingface.py --repo-name your-actual-username/ticketcat-distilbert
```

### 3. Test It Works

```bash
python src/models/load_from_huggingface.py --demo
```

That's it! Your model is now publicly available and anyone can load it with:

```python
from src.models.load_from_huggingface import HuggingFaceTicketClassifier

classifier = HuggingFaceTicketClassifier()  # Loads from your repo automatically
category, confidence = classifier.predict("Can't login to account")
```

---

## For Model Users

If the default model is already configured, you can simply:

```python
from src.models.load_from_huggingface import HuggingFaceTicketClassifier

# Load the default pre-trained model
classifier = HuggingFaceTicketClassifier()

# Classify tickets
category, confidence = classifier.predict("System is very slow")
print(f"{category} ({confidence:.2%})")
```

Or use a different model:

```python
classifier = HuggingFaceTicketClassifier('another-user/their-model')
```

