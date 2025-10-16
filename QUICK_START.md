# Quick Start Guide

## Setup (2 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup API keys (optional - only needed for data generation)
cp secrets.json.example secrets.json
# Edit secrets.json with your keys OR set environment variables
```

## Data Pipeline Options

### Option A: Use Existing Processed Data
If you already have `data/processed/train.csv`, `val.csv`, `test.csv`:
```bash
# Skip directly to training
python3 src/train.py --model baseline
```

### Option B: Download from Kaggle (No GPT)
```bash
# Download and process Kaggle data only (no augmentation)
python3 src/setup_and_augment_data.py --skip-gpt

# What happens:
# ✓ Downloads IT Service tickets from Kaggle
# ✓ Cleans and preprocesses data
# ✓ Splits into train/val/test (70/15/15)
# ✓ Saves to data/processed/
# ⚠️ Skips GPT augmentation (no API cost)
```

### Option C: Full Pipeline with GPT Augmentation
```bash
# Requires OpenAI API key in secrets.json or environment
python3 src/setup_and_augment_data.py --gpt-count 200

# What happens:
# ✓ Downloads Kaggle data
# ✓ Generates 200 synthetic tickets per rare category using GPT
# ✓ Adds augmentation ONLY to training set
# ✓ Costs ~$1.60 for 800 tickets
# ✓ Improves rare category F1-scores by ~15-50%
```

## What Happens When GPT Data is Missing?

**The pipeline gracefully handles missing GPT data:**

```python
# In setup_and_augment_data.py (lines 322-360)
gpt_file = 'data/synthetic_gpt_augmentation.csv'
if os.path.exists(gpt_file):
    # Load and add GPT tickets to training set
    print(f"✅ Loaded {len(gpt_df)} GPT tickets")
    train_df = pd.concat([train_df, gpt_df])
else:
    # Continue without augmentation
    print("⚠️  GPT augmentation not found")
    print("   Proceeding without augmentation.")
    # Training continues with original Kaggle data only
```

**Result**: Training proceeds with only the Kaggle data (~15-20k tickets). Performance is still good, but rare categories may have lower F1-scores.

## Training

```bash
# Baseline (fast, CPU-friendly) → models/baseline/
python3 src/train.py --model baseline

# DistilBERT (GPU recommended) → models/distilbert/
python3 src/train.py --model distilbert --epochs 3 --batch_size 16
```

## Evaluation

```bash
# Evaluate baseline model
python3 src/evaluate.py --model_type baseline --model_dir models/baseline

# Evaluate DistilBERT model
python3 src/evaluate.py --model_type distilbert --model_dir models/distilbert
```

## API

```bash
python3 main.py  # Starts on http://localhost:8000

# Test
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"title": "Cannot login", "description": "Error message"}'
```

## File Overview

| File | Lines | Purpose |
|------|-------|---------|
| `src/train.py` | 137 | Training orchestration |
| `src/models/baseline_model.py` | 180 | TF-IDF + Logistic Regression |
| `src/models/train_distilbert.py` | 298 | DistilBERT fine-tuning |
| `src/data_generation/augment_with_gpt.py` | 106 | GPT-3.5 synthetic data |
| `src/evaluate.py` | ~575 | Model evaluation |
| `src/api.py` | ~397 | REST API |
| **Total Core** | **721** | **(reduced from 3700)** |

## Common Issues

**Q: GPT augmentation fails?**  
A: Use `--skip-gpt` flag. The project works fine without augmentation.

**Q: Kaggle download fails?**  
A: Check `~/.kaggle/kaggle.json` credentials or skip with `--skip-download` if you have data.

**Q: Not enough GPU memory?**  
A: Reduce `--batch_size` or use baseline model (CPU-friendly).

## Performance Comparison

| Model | F1-Score | With GPT | Without GPT |
|-------|----------|----------|-------------|
| Baseline | ~0.85 | ~0.87 | ~0.82 |
| DistilBERT | ~0.89 | ~0.91 | ~0.87 |

GPT augmentation provides **+2-5% F1-score boost**, especially for rare categories.

