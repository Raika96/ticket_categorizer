# Model Location Update

## Summary

Updated the project to save and load the final DistilBERT model from a standardized location, making it easier to use retrained models without manual intervention.

## Changes Made

### 1. Updated Model Save Location
**File: `src/models/train_distilbert.py`**
- Changed default save location from `models/distilbert` to `models/distilbert/final`
- Changed default load location from `models/distilbert` to `models/distilbert/final`

### 2. Updated Training Script
**File: `src/train.py`**
- DistilBERT models now save to `models/distilbert/final` by default
- Checkpoints still save to `models/distilbert/checkpoints/`

### 3. Updated API Inference
**File: `src/infer.py`**
- API now loads from `models/distilbert/final` instead of a specific checkpoint
- Automatically picks up newly trained models after API restart

### 4. Migrated Current Model
- Copied existing checkpoint-6195 to `models/distilbert/final/`
- API immediately works with the new location

## New Workflow

### Training a New Model
```bash
# Train DistilBERT (saves to models/distilbert/final/)
python3 src/train.py --model distilbert --epochs 3

# Restart API to use new model
lsof -ti:8000 | xargs kill -9
python3 main.py --model distilbert
```

### Directory Structure
```
models/distilbert/
├── final/                    # Production model (used by API)
│   ├── model.safetensors
│   ├── config.json
│   ├── classifier_config.json
│   ├── vocab.txt
│   └── tokenizer files...
└── checkpoints/              # Training checkpoints
    ├── checkpoint-2000/
    ├── checkpoint-4000/
    └── checkpoint-6195/
```

## Benefits

1. **Automatic Updates**: Retraining saves directly to the API's model location
2. **Simple Deployment**: Just restart the API after training
3. **Clear Separation**: Final model vs. intermediate checkpoints
4. **Consistent Paths**: All scripts use the same default location

## Backward Compatibility

Old checkpoints in `models/distilbert/checkpoints/` are preserved and can still be loaded manually:

```bash
# Evaluate a specific checkpoint
python3 src/evaluate.py --model_type distilbert --model_dir models/distilbert/checkpoints/checkpoint-6195
```
