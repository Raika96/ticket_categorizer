# Model Directory Structure

## Changes Made

**Before**: Models saved with mixed naming conventions in a single `models/` directory  
**After**: Clean separation with consistent naming in dedicated subdirectories

### Directory Layout

```
models/
├── baseline/                  # Baseline model (TF-IDF + Logistic Regression)
│   ├── model.joblib          # Trained LogisticRegression model
│   ├── vectorizer.joblib     # Fitted TF-IDF vectorizer
│   ├── config.json           # Model configuration & metadata
│   └── validation_metrics.json
│
└── distilbert/               # DistilBERT model
    ├── pytorch_model.bin     # Trained model weights
    ├── config.json           # HuggingFace model config
    ├── classifier_config.json # Our classifier config & metadata
    ├── vocab.txt             # Tokenizer vocabulary
    ├── tokenizer_config.json # Tokenizer configuration
    ├── special_tokens_map.json
    ├── validation_metrics.json
    └── checkpoints/          # Training checkpoints (auto-cleaned)
        ├── checkpoint-500/
        └── checkpoint-1000/
```

## Usage

### Training
```bash
# Baseline → models/baseline/
python3 src/train.py --model baseline

# DistilBERT → models/distilbert/
python3 src/train.py --model distilbert

# Custom output directory
python3 src/train.py --model baseline --output_dir experiments/run1
```

### Evaluation
```bash
# Evaluate baseline
python3 src/evaluate.py --model_type baseline --model_dir models/baseline

# Evaluate DistilBERT
python3 src/evaluate.py --model_type distilbert --model_dir models/distilbert
```

### Loading Models in Code

**Baseline:**
```python
from src.models.baseline_model import BaselineTicketClassifier

# Load model
classifier = BaselineTicketClassifier.load('models/baseline')

# Predict
predictions, confidences = classifier.predict(["Cannot login", "Slow performance"])
```

**DistilBERT:**
```python
from src.models.train_distilbert import DistilBERTClassifier

# Load model
classifier = DistilBERTClassifier.load('models/distilbert')

# Predict
predictions, confidences = classifier.predict(["Cannot login", "Slow performance"])
```

## Benefits

✅ **Clean Separation**: Each model type has its own directory  
✅ **Consistent Naming**: `config.json`, `model.joblib`, etc.  
✅ **No Conflicts**: Can have both models simultaneously  
✅ **Easy Comparison**: Side-by-side model evaluation  
✅ **Version Control**: Can save multiple versions in separate dirs

## Migration from Old Structure

If you have old models with the previous naming convention:

```bash
# Old structure
models/
├── baseline_model.joblib
├── baseline_vectorizer.joblib
├── baseline_config.json
├── distilbert_model/
└── distilbert_config.json

# Simply retrain or manually reorganize:
mkdir -p models/baseline models/distilbert
mv baseline_*.joblib models/baseline/
mv baseline_config.json models/baseline/config.json
mv distilbert_model/* models/distilbert/
```

## File Size Reference

| Model | Size | Description |
|-------|------|-------------|
| Baseline | ~10 MB | Vectorizer vocabulary + model weights |
| DistilBERT | ~250 MB | Pre-trained transformer + fine-tuned weights |

---

**Note**: The `checkpoints/` subdirectory in DistilBERT stores intermediate training checkpoints. Only the best model (based on F1-score) is saved to the main directory.
