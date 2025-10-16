# Support Ticket Classification

Automated classification of customer support tickets into 9 predefined categories using machine learning.

## Overview

This project implements a ticket classification system with two models:
- **Baseline**: TF-IDF + Logistic Regression (fast, CPU-friendly)
- **DistilBERT**: Transformer-based model (higher accuracy, GPU recommended)

Both models support confidence thresholding to flag uncertain predictions for manual review.

## Categories

1. Account Access / Login Issues
2. Billing & Payments
3. Bug / Defect Reports
4. Feature Requests
5. General Inquiries / Other
6. How-To / Product Usage Questions
7. Integration Issues
8. Performance Problems
9. Security & Compliance

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Edit secrets.json with your OpenAI and Kaggle API keys
```

## Core Processes

### 1. Data Processing

The data pipeline downloads and processes IT service tickets from Kaggle, applies cleaning and preprocessing, and creates train/validation/test splits.

```bash
# Run complete data pipeline
python3 src/setup_and_augment_data.py

# Options:
# --use-existing: Reuse existing processed data
# --gpt-count: Number of synthetic tickets per rare category (default: 20)
```

**Pipeline Steps:**
1. Download IT Service ticket dataset from Kaggle
2. Clean and normalize text (PII redaction, whitespace, special characters)
3. Map original categories to 9 target categories
4. Create stratified 70/15/15 train/val/test split
5. Optionally generate synthetic data for rare categories using GPT-3.5
6. Save processed datasets to `data/processed/`

**Output Files:**
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/label_mapping.json`

#### GPT Data Augmentation

To address class imbalance, we use GPT-3.5-Turbo to generate synthetic training samples for underrepresented categories. This improves model performance on rare classes without collecting additional real data.

**Key Points:**
- Applied only to 4 rare categories: Security & Compliance, Performance Problems, Integration Issues, General Inquiries
- Default: 20 synthetic tickets per rare category (configurable via `--gpt-count`)
- Synthetic data added **only to training set** after the initial train/val/test split to prevent data leakage
- Generated tickets are cached and reused to avoid redundant API calls

### 2. Model Training

#### Baseline Model (TF-IDF + Logistic Regression)

Fast training on CPU. Uses TF-IDF vectorization with class weighting to handle imbalanced data.

```bash
python3 src/train.py --model baseline --confidence_threshold 0.6 --seed 42
```

**Training Process:**
- Vectorizes text using TF-IDF (max 5000 features)
- Trains Logistic Regression with balanced class weights
- Validates on validation set
- Saves model to `models/baseline/`

**Training Time:** ~1 minute on CPU

#### DistilBERT Model

Transformer-based model for higher accuracy. Requires GPU for practical training times.

```bash
python3 src/train.py --model distilbert --epochs 3 --batch_size 16 --seed 42
```

**Training Process:**
- Fine-tunes DistilBERT-base-uncased on ticket data
- Uses AdamW optimizer with learning rate 2e-5
- Applies early stopping (patience=3 epochs)
- Saves checkpoints to `models/distilbert/checkpoints/`
- Final model saved to `models/distilbert/`

**Training Time:** 
- CPU: ~76 minutes (3 epochs)
- GPU: ~10-15 minutes (3 epochs)

**Training Arguments:**
```bash
--epochs 3                      # Number of training epochs
--batch_size 16                 # Batch size for training
--learning_rate 2e-5           # Learning rate
--confidence_threshold 0.5     # Threshold for flagging uncertain predictions
--early_stopping_patience 3    # Stop if no improvement for N epochs
--seed 42                      # Random seed for reproducibility
```

### 3. Load Pre-trained Model from Hugging Face

Download the pre-trained model from Hugging Face Hub to your local directory:

```bash
python3 src/models/load_from_huggingface.py --save-local models/distilbert/final
```

### 4. Model Evaluation

Evaluate models on validation or test sets with comprehensive metrics and error analysis.

```bash
# Evaluate baseline model
python3 src/evaluate.py --model_type baseline --model_dir models/baseline

# Evaluate DistilBERT model
python3 src/evaluate.py --model_type distilbert --model_dir models/distilbert/final

# Use test set (only do this once at the end)
python3 src/evaluate.py --model_type distilbert --model_dir models/distilbert/final --use_test_set
```

**Evaluation Process:**
1. Loads trained model and test/validation data
2. Generates predictions with confidence scores
3. Computes performance metrics
4. Performs error analysis
5. Generates visualizations (confusion matrix, confidence distribution)
6. Saves results to `experiments/`

**Output Files:**
- `experiments/confusion_matrix_{split}.png`: Confusion matrix visualization
- `experiments/confidence_analysis.png`: Confidence distribution plot
- `experiments/error_examples.csv`: Sample of misclassified tickets
- `experiments/evaluation_results_{split}.json`: Detailed metrics
- `experiments/classification_report_{split}.txt`: Per-class performance
- `experiments/evaluation_summary.txt`: Human-readable evaluation summary

*Note: `{split}` is either `val` or `test` depending on which dataset was evaluated.*

### 5. Command-Line Inference

Classify tickets directly from the command line without running the full API server.

```bash
# Quick prediction
python3 src/infer.py '{"title": "Cannot login", "description": "Password not working"}'

# From stdin
echo '{"title": "Refund request", "description": "Want my money back"}' | python3 src/infer.py

# Use baseline model or compact output
python3 src/infer.py --model baseline --compact '{"title": "Bug report", "description": "App crashes"}'
```

### 6. API Server

Run the classification API for real-time predictions.

```bash
# Start API with DistilBERT model
python3 main.py --model distilbert

# Start with baseline model
python3 main.py --model baseline

# Custom host/port
python3 main.py --host 0.0.0.0 --port 5000
```

**API Endpoints:**
- `GET /`: API information
- `GET /health`: Health check
- `POST /classify`: Classify a single ticket
- `POST /classify/batch`: Classify multiple tickets
- `GET /stats`: API usage statistics
- `GET /docs`: Interactive API documentation (Swagger UI)

**Example Request:**
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Cannot login to my account",
    "description": "I am getting an error when trying to access my account"
  }'
```

**Example Response:**
```json
{
  "predicted_category": "Account Access / Login Issues",
  "confidence": 0.9412,
  "needs_manual_review": false,
  "model_type": "distilbert",
  "timestamp": "2025-10-16T14:00:00.000000"
}
```

## Evaluation Metrics

### Primary Metrics

- **Accuracy**: Percentage of correct predictions
- **F1-Score (Weighted)**: Harmonic mean of precision and recall, weighted by class size
- **F1-Score (Macro)**: Unweighted average F1 across all classes (better for imbalanced data)
- **Cohen's Kappa**: Agreement metric accounting for chance (0-1 scale, >0.8 is excellent)
- **Balanced Accuracy**: Average recall per class (handles class imbalance)

### Confidence Analysis

- **Mean Confidence**: Average prediction confidence across all samples
- **High Confidence Rate**: Percentage of predictions above threshold (default: 0.6)
- **Accuracy (High Confidence)**: Accuracy on high-confidence predictions only
- **Auto-Assignment Rate**: Percentage of tickets that can be auto-assigned (high confidence)

### Per-Class Metrics

- **Precision**: Of predictions for a class, how many were correct
- **Recall**: Of actual class members, how many were predicted correctly
- **F1-Score**: Harmonic mean of precision and recall per class
- **Support**: Number of samples in each class

## Test Set Results

Final evaluation on held-out test set (7,036 tickets):

### Overall Performance

| Metric | Baseline | DistilBERT | Improvement |
|--------|----------|------------|-------------|
| Accuracy | 76.88% | 92.38% | +15.50% |
| F1-Score (Weighted) | 77.10% | 92.36% | +15.26% |
| F1-Score (Macro) | 71.41% | 90.46% | +19.05% |
| Cohen's Kappa | 0.7215 | 0.9072 | +25.74% |
| Balanced Accuracy | 74.76% | 89.66% | +14.90% |

### Confidence Metrics

| Metric | Baseline | DistilBERT |
|--------|----------|------------|
| Mean Confidence | 61.80% | 96.87% |
| Auto-Assignment Rate | 51.05% | 98.01% |
| Accuracy (High Confidence) | 89.39% | 93.43% |

### Per-Category F1-Scores

| Category | Baseline | DistilBERT | Delta |
|----------|----------|------------|-------|
| General Inquiries / Other | 82.65% | 96.72% | +14.07% |
| Account Access / Login Issues | 85.27% | 93.42% | +8.15% |
| Bug / Defect Reports | 76.02% | 92.62% | +16.60% |
| Feature Requests | 71.31% | 91.69% | +20.38% |
| Performance Problems | 68.92% | 91.88% | +22.96% |
| Security & Compliance | 64.86% | 88.17% | +23.31% |
| Billing & Payments | 70.44% | 86.45% | +16.01% |
| How-To / Product Usage | 65.77% | 87.94% | +22.17% |
| Integration Issues | 57.48% | 85.21% | +27.73% |

### Error Analysis

**DistilBERT:**
- Total errors: 536 / 7,036 (7.62%)
- Most common misclassifications:
  - Bug Reports → Account Access (42 errors)
  - Feature Requests → Account Access (30 errors)
  - How-To Questions → Account Access (28 errors)

**Baseline:**
- Total errors: 1,627 / 7,036 (23.12%)
- Most common misclassifications:
  - Account Access → Bug Reports (114 errors)
  - Account Access → Feature Requests (89 errors)
  - How-To Questions → General Inquiries (70 errors)

## Technical Details

### Text Preprocessing

The `TextCleaner` class handles:
- PII redaction (emails, phone numbers, IDs)
- Whitespace normalization
- Special character handling
- Extreme case detection (very short/long tickets)
- Duplicate detection

### Class Imbalance Handling

- **Training**: Class-weighted loss functions
- **Baseline**: `class_weight='balanced'` in Logistic Regression
- **DistilBERT**: Weighted cross-entropy loss
- **Data Augmentation**: Optional GPT-3.5 generation for rare categories

### Confidence Thresholding

Predictions below the confidence threshold (default: 0.6 for baseline, 0.5 for DistilBERT) are flagged for manual review:
- `needs_manual_review: true`
- Lower threshold = more automation, potentially more errors
- Higher threshold = more manual review, fewer errors

### Model Persistence

**Baseline Model Files:**
- `models/baseline/model.joblib`: Trained Logistic Regression
- `models/baseline/vectorizer.joblib`: Fitted TF-IDF vectorizer
- `models/baseline/config.json`: Model configuration and metadata

**DistilBERT Model Files:**
- `models/distilbert/final/model.safetensors`: Model weights
- `models/distilbert/final/config.json`: HuggingFace model config
- `models/distilbert/final/classifier_config.json`: Custom config and metadata
- `models/distilbert/final/vocab.txt`, `tokenizer_config.json`: Tokenizer files
- `models/distilbert/checkpoints/`: Training checkpoints

**Hugging Face Integration:**
- `src/models/push_to_huggingface.py`: Upload model to Hugging Face Hub
- `src/models/load_from_huggingface.py`: Load model from Hugging Face Hub

## Project Structure

```
TicketCat/
├── src/
│   ├── models/
│   │   ├── baseline_model.py            # TF-IDF + Logistic Regression
│   │   ├── train_distilbert.py          # DistilBERT fine-tuning
│   │   ├── push_to_huggingface.py       # Upload model to HF Hub
│   │   ├── load_from_huggingface.py     # Load model from HF Hub
│   │   └── README.md                    # Model documentation
│   ├── preprocessing/
│   │   └── text_cleaner.py          # Text cleaning and PII redaction
│   ├── data_generation/
│   │   ├── it_service_processor.py  # Kaggle data processing
│   │   └── augment_with_gpt.py      # GPT data augmentation
│   ├── inference/
│   │   └── gpt_fallback.py          # GPT fallback for low confidence
│   ├── train.py                      # Model training script
│   ├── evaluate.py                   # Model evaluation script
│   ├── infer.py                      # Inference utilities
│   ├── api.py                        # FastAPI application
│   └── setup_and_augment_data.py    # Data pipeline
├── data/
│   ├── processed/                    # Train/val/test splits
│   └── external/                     # Downloaded raw data
├── models/
│   ├── baseline/                     # Baseline model artifacts
│   └── distilbert/                   # DistilBERT model artifacts
├── experiments/                      # Evaluation results
├── examples/
│   └── huggingface_workflow_example.py  # HF integration example
├── main.py                           # API entry point
├── load_secrets.py                   # API key management
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+ (for DistilBERT)
- transformers 4.30+
- scikit-learn 1.2+
- pandas, numpy
- fastapi, uvicorn (for API)
- GPU recommended for DistilBERT training (CPU works but slow)

## Limitations

**Model & Data:**
- **Domain-Specific**: Trained on IT service tickets; performance may degrade on other domains (healthcare, legal, e-commerce)
- **Category Overlap**: Some categories have overlapping language (e.g., Bug Reports vs. Feature Requests), leading to confusion on short or ambiguous tickets
- **Synthetic Data**: GPT-3.5-generated tickets improve balance but may not fully reflect real-world edge cases
- **English Only**: Optimized for English; non-English or code-mixed tickets may not classify accurately

**Technical:**
- **Resource Requirements**: DistilBERT requires GPU for training and is slower than baseline on CPU-only machines
- **Static Categories**: Only 9 predefined categories; adding new ones requires retraining
- **No Active Learning**: Misclassified tickets flagged for review but not automatically fed back for model improvement

**Operational:**
- **Confidence Threshold**: Fixed threshold (0.5-0.6) may not be optimal for all environments
- **Limited Explainability**: No detailed explanations for predictions
- **PII Handling**: Redacts common patterns but may not catch all sensitive information

## Future Improvements

**Model Enhancements:**
- **Additional Data Sources**: Integrate more real-world support data to improve generalization and edge case coverage
- **Advanced Architectures**: Explore RoBERTa, DeBERTa, or domain-adapted LLMs for higher accuracy on overlapping categories
- **Multilingual Support**: Fine-tune multilingual models (XLM-Roberta) or add translation pipelines for non-English tickets
- **Explainability**: Integrate LIME/SHAP or attention visualization to explain predictions

**Data & Feedback:**
- **Active Learning**: Build feedback loop to add misclassified tickets back into training set
- **Fine-Grained Categories**: Add subcategories (e.g., "Password Reset" vs. "MFA Issues" under Account Access)
- **Data Quality Monitoring**: Automate drift detection and retraining triggers

**Deployment:**
- **Production-Ready API**: Containerize with Docker/Kubernetes, add load balancing and monitoring
- **Batch Processing**: Support async processing and message queues (Kafka, SQS) for large-scale deployments
- **Model Versioning**: Implement A/B testing and version control for safe production updates

**Operations:**
- **Dynamic Thresholds**: Allow per-category or adaptive confidence thresholds
- **Review Dashboard**: Web UI for agents to review and correct low-confidence predictions

---

## Disclaimer

AI assistants were used during the development and documentation of this project. All generated code and documentation have been reviewed and validated by the project author to ensure correctness, quality, and adherence to best practices.
