#!/usr/bin/env python3
"""
End-to-End Data Pipeline for Ticket Classification

Proper ML Pipeline Workflow (prevents data leakage):
1. Download IT Service tickets from Kaggle
2. Process and clean the raw data
3. Split into train/val/test sets (70/15/15)
4. Generate GPT augmentation for rare categories
5. Add augmentation ONLY to training set
6. Save final train/val/test splits

Key Feature: Augmentation only applied to training data to prevent data leakage.
Validation and test sets remain pure for proper evaluation.

Usage:
    # Full pipeline (download + augment)
    python src/setup_and_augment_data.py --gpt-count 200
    
    # Skip Kaggle download (use existing data)
    python src/setup_and_augment_data.py --skip-download
    
    # Skip GPT augmentation (faster, no API cost)
    python src/setup_and_augment_data.py --skip-gpt
    
    # Use existing processed data (just verify it exists)
    python src/setup_and_augment_data.py --use-existing
    
    Or from project root:
    python -m src.setup_and_augment_data [OPTIONS]
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Change to project root to ensure all relative paths work correctly
os.chdir(project_root)

# Import secrets loader from root
from load_secrets import load_secrets, setup_environment, check_secrets


def print_section(title, emoji=""):
    """Print a formatted section header"""
    print("\n" + "="*100)
    print(f" {emoji} {title}")
    print("="*100 + "\n")


def run_command(cmd, description, check=True):
    """Run a shell command and handle errors"""
    print(f"  {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f" {description} - Success!")
            if result.stdout.strip():
                print(result.stdout)
            return True
        else:
            print(f" {description} - Failed!")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f" {description} - Failed with error:")
        print(e.stderr)
        return False


def check_kaggle_credentials():
    """Check if Kaggle credentials are configured"""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists():
        print(" Kaggle credentials not found!")
        print("\n To set up Kaggle API:")
        print("   1. Go to https://www.kaggle.com/account")
        print("   2. Click 'Create New API Token'")
        print("   3. Move kaggle.json to ~/.kaggle/")
        print("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("\nSee KAGGLE_SETUP.md for detailed instructions.")
        return False
    return True


def download_kaggle_data():
    """Download IT Service dataset from Kaggle"""
    print_section("Step 1: Download IT Service Tickets from Kaggle", "")
    
    # Check credentials
    if not check_kaggle_credentials():
        print("\n  Skipping download. Please set up Kaggle credentials first.")
        return False
    
    # Create data directory
    os.makedirs('data/external', exist_ok=True)
    
    # Download dataset
    dataset = "adisongoh/it-service-ticket-classification-dataset"
    success = run_command(
        f'kaggle datasets download -d {dataset} -p data/external --unzip',
        f"Downloading {dataset}"
    )
    
    if not success:
        print("\n Download failed. Dataset might already exist or credentials are invalid.")
        return False
    
    print(f"\n Dataset downloaded to: data/external/")
    return True


def process_it_tickets():
    """Process IT Service tickets using the processor"""
    print_section("Step 2: Process IT Service Tickets", "")
    
    # Check if processed file already exists
    output_file = 'data/processed_it_tickets.csv'
    if os.path.exists(output_file):
        print(f" Processed tickets already exist: {output_file}")
        import pandas as pd
        df = pd.read_csv(output_file)
        print(f"    {len(df):,} tickets loaded")
        return True
    
    # Check if raw data exists (try multiple possible filenames)
    possible_files = [
        'data/external/all_tickets_processed_improved_v3.csv',
        'data/external/IT Service - Ticket Classification Dataset.csv'
    ]
    
    raw_file = None
    for f in possible_files:
        if os.path.exists(f):
            raw_file = f
            print(f" Found raw data: {f}")
            break
    
    if not raw_file:
        print(f" Raw data file not found. Checked:")
        for f in possible_files:
            print(f"   - {f}")
        print("\n   Make sure the Kaggle download completed successfully.")
        print("\n Alternative: If you already have processed data in data/processed/,")
        print("   you can skip directly to training with that data.")
        return False
    
    # Run processor
    cmd = 'python3 src/data_generation/it_service_processor.py'
    success = run_command(cmd, "Processing IT tickets")
    
    if success:
        if os.path.exists(output_file):
            print(f"\n Processed tickets saved to: {output_file}")
            return True
    
    return False


def generate_gpt_augmentation(count_per_category=200):
    """Generate GPT augmentation for rare categories"""
    print_section(f"Step 3: Generate GPT Augmentation ({count_per_category} per category)", "")
    
    # Check if GPT augmentation already exists
    output_file = 'data/synthetic_gpt_augmentation.csv'
    if os.path.exists(output_file):
        import pandas as pd
        try:
            existing_df = pd.read_csv(output_file)
            print(f" GPT augmentation already exists: {output_file}")
            print(f"   Contains {len(existing_df):,} tickets")
            print("\n Using existing GPT data (no API calls needed)")
            print("   To regenerate, delete the file and run again.")
            return True
        except Exception as e:
            print(f"  Found {output_file} but couldn't read it: {e}")
            print("   Will regenerate...")
    
    # Check for OpenAI API key
    status = check_secrets()
    if not status['openai_ok']:
        print(" OpenAI API key not configured!")
        print("\n To configure:")
        print("   1. Copy secrets.json.example to secrets.json")
        print("   2. Add your OpenAI API key")
        print("   OR")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("\n  Skipping GPT augmentation.")
        return False
    
    print(f" OpenAI API key found: {status['openai_key']}")
    
    # Run GPT augmentation
    cmd = f'python3 src/data_generation/augment_with_gpt.py --yes --count {count_per_category}'
    success = run_command(cmd, f"Generating {count_per_category * 4} GPT tickets")
    
    if success:
        if os.path.exists(output_file):
            print(f"\n GPT augmentation saved to: {output_file}")
            return True
    
    return False


def merge_and_preprocess():
    """
    Process and split data properly:
    1. Load and clean Kaggle IT tickets
    2. Split into train/val/test (70/15/15)
    3. Add GPT augmentation ONLY to training set
    4. Save final splits
    """
    print_section("Step 4: Process and Split Data", "")
    
    import pandas as pd
    from src.preprocessing.text_cleaner import TextCleaner, handle_duplicates, handle_extreme_cases
    from sklearn.model_selection import train_test_split
    
    # ========== STEP 1: Load and Clean IT Tickets ==========
    it_file = 'data/processed_it_tickets.csv'
    if not os.path.exists(it_file):
        print(f" IT tickets not found: {it_file}")
        return False
    
    print(f" Loading IT tickets from: {it_file}")
    df = pd.read_csv(it_file)
    print(f"    Loaded {len(df):,} IT tickets")
    
    # Show initial category distribution
    print("\n Initial Category Distribution:")
    for category, count in df['category'].value_counts().sort_index().items():
        print(f"   {category:35} {count:6,}")
    
    # Handle duplicates in original data
    print("\n Removing duplicates from IT tickets...")
    initial_count = len(df)
    df, n_duplicates = handle_duplicates(df, text_cols=['title', 'description'], keep='first')
    print(f"    Removed {n_duplicates:,} duplicates ({initial_count:,} → {len(df):,})")
    
    # Clean text
    print("\n Cleaning text...")
    cleaner = TextCleaner(min_length=10, max_length=2000, redact_pii=True)
    
    for col in ['title', 'description']:
        if f'{col}_clean' not in df.columns:
            df[f'{col}_clean'] = df[col].apply(cleaner.clean_text)
    
    # Create combined text
    if 'text_clean' not in df.columns:
        df['text_clean'] = df['title_clean'] + ' ' + df['description_clean']
    
    # Remove short texts
    df = df[df['text_clean'].str.len() >= 10].copy()
    print(f"    Cleaned and filtered: {len(df):,} tickets remain")
    
    # Handle extreme cases
    print("\n Handling extreme cases...")
    df['combined_length'] = df['text_clean'].str.len()
    df, extreme_stats = handle_extreme_cases(
        df,
        length_col='combined_length',
        min_percentile=1,
        max_percentile=99
    )
    
    # Create label column and mapping
    categories = sorted(df['category'].unique())
    label_map = {cat: idx for idx, cat in enumerate(categories)}
    df['label'] = df['category'].map(label_map)
    
    # Save label mapping in the correct format
    os.makedirs('data/processed', exist_ok=True)
    label_mapping = {
        'label_to_id': label_map,
        'id_to_category': {str(idx): cat for cat, idx in label_map.items()},
        'num_classes': len(categories)
    }
    with open('data/processed/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"\n Saved label mapping to: data/processed/label_mapping.json")
    
    # ========== STEP 2: Split into Train/Val/Test ==========
    print("\n  Splitting data (70% train, 15% val, 15% test)...")
    
    # First split: train vs temp (test+val)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df['category']
    )
    
    # Second split: val vs test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['category']
    )
    
    print(f"    Train: {len(train_df):,} tickets")
    print(f"    Val:   {len(val_df):,} tickets")
    print(f"    Test:  {len(test_df):,} tickets")
    
    # ========== STEP 3: Add GPT Augmentation to TRAIN ONLY ==========
    gpt_file = 'data/synthetic_gpt_augmentation.csv'
    gpt_tickets_added = 0
    
    if os.path.exists(gpt_file):
        print(f"\n Loading GPT augmentation from: {gpt_file}")
        gpt_df = pd.read_csv(gpt_file)
        print(f"    Loaded {len(gpt_df):,} GPT tickets")
        
        # Clean GPT data
        print("\n Cleaning GPT augmentation...")
        for col in ['title', 'description']:
            if f'{col}_clean' not in gpt_df.columns:
                gpt_df[f'{col}_clean'] = gpt_df[col].apply(cleaner.clean_text)
        
        if 'text_clean' not in gpt_df.columns:
            gpt_df['text_clean'] = gpt_df['title_clean'] + ' ' + gpt_df['description_clean']
        
        # Add label if not present
        if 'label' not in gpt_df.columns:
            gpt_df['label'] = gpt_df['category'].map(label_map)
        
        # Get common columns between train and GPT data
        common_cols = list(set(train_df.columns) & set(gpt_df.columns))
        
        # Add GPT data ONLY to training set
        print(f"\n Adding GPT augmentation to TRAINING SET ONLY...")
        print(f"   Before augmentation: {len(train_df):,} training tickets")
        train_df = pd.concat([train_df[common_cols], gpt_df[common_cols]], ignore_index=True)
        gpt_tickets_added = len(gpt_df)
        print(f"   After augmentation:  {len(train_df):,} training tickets (+{gpt_tickets_added:,})")
        
        # Remove duplicates after augmentation
        train_df, n_aug_duplicates = handle_duplicates(train_df, text_cols=['title', 'description'], keep='first')
        if n_aug_duplicates > 0:
            print(f"    Removed {n_aug_duplicates:,} duplicate augmentations")
        
    else:
        print(f"\n  GPT augmentation not found: {gpt_file}")
        print("   Proceeding without augmentation.")
    
    # ========== STEP 4: Show Final Statistics ==========
    print("\n FINAL DATA SPLIT:")
    print(f"   Train: {len(train_df):,} tickets (includes {gpt_tickets_added:,} augmented)")
    print(f"   Val:   {len(val_df):,} tickets (original data only)")
    print(f"   Test:  {len(test_df):,} tickets (original data only)")
    
    print("\n Training Set Category Distribution:")
    for category, count in train_df['category'].value_counts().sort_index().items():
        print(f"   {category:35} {count:6,}")
    
    # ========== STEP 5: Save Final Splits ==========
    print("\n Saving processed splits...")
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    print(f"    Saved to: data/processed/")
    
    # Save augmented version separately for reference
    if gpt_tickets_added > 0:
        os.makedirs('data/processed_augmented', exist_ok=True)
        train_df.to_csv('data/processed_augmented/train_augmented.csv', index=False)
        val_df.to_csv('data/processed_augmented/val.csv', index=False)
        test_df.to_csv('data/processed_augmented/test.csv', index=False)
        print(f"    Also saved augmented version to: data/processed_augmented/")
    
    # Save statistics
    stats = {
        'timestamp': datetime.now().isoformat(),
        'original_tickets': len(df),
        'gpt_augmentation_tickets': gpt_tickets_added,
        'train_tickets': len(train_df),
        'train_original_tickets': len(train_df) - gpt_tickets_added,
        'train_augmented_tickets': gpt_tickets_added,
        'val_tickets': len(val_df),
        'test_tickets': len(test_df),
        'duplicates_removed': n_duplicates,
        'extreme_cases_removed': extreme_stats['removed'],
        'train_category_distribution': train_df['category'].value_counts().to_dict(),
        'val_category_distribution': val_df['category'].value_counts().to_dict(),
        'test_category_distribution': test_df['category'].value_counts().to_dict(),
        'augmentation_applied': gpt_tickets_added > 0
    }
    
    with open('data/processed/preprocessing_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"    Saved statistics to: data/processed/preprocessing_stats.json")
    
    print("\n" + "="*100)
    print(" DATA PROCESSING COMPLETE!")
    print("="*100)
    print("\n Key Points:")
    print(f"   • Original Kaggle data split into train/val/test")
    print(f"   • GPT augmentation added ONLY to training set (no data leakage)")
    print(f"   • Validation and test sets remain pure/unseen")
    print(f"   • Ready for model training!")
    
    return True


def print_summary(args, start_time):
    """Print pipeline summary"""
    duration = time.time() - start_time
    
    print_section("Pipeline Complete! ", "")
    
    print(f"  Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print("\n Output files:")
    print("    data/processed/train.csv (includes augmentation)")
    print("    data/processed/val.csv (original data only)")
    print("    data/processed/test.csv (original data only)")
    print("    data/processed/label_mapping.json")
    print("    data/processed/preprocessing_stats.json")
    
    if os.path.exists('data/processed_augmented/train_augmented.csv'):
        print("\n Augmented version also saved to:")
        print("    data/processed_augmented/train_augmented.csv")
        print("    data/processed_augmented/val.csv")
        print("    data/processed_augmented/test.csv")
    
    print("\n Data Pipeline Summary:")
    print("   • Downloaded and processed Kaggle IT Service tickets")
    print("   • Split into train/val/test (70/15/15) with stratification")
    print("   • GPT augmentation added ONLY to training set")
    print("   • Validation and test sets remain clean (no augmentation)")
    print("   • This prevents data leakage and ensures valid evaluation")
    
    print("\n Next Steps:")
    print("   1. Train baseline model:")
    print("      python3 src/train.py --model baseline")
    print("\n   2. Train DistilBERT model (on Colab with GPU):")
    print("      - Upload data files to Colab")
    print("      - Run: !python train_distilbert_colab.py")
    print("\n   3. Evaluate models:")
    print("      python3 src/evaluate.py --model_type baseline")
    print("      python3 src/evaluate.py --model_type distilbert")
    print("\n   4. Start the API:")
    print("      python3 main.py")
    
    print("\n" + "="*100)


def main():
    """Main pipeline"""
    parser = argparse.ArgumentParser(
        description='End-to-end data pipeline: Download, augment, and preprocess IT tickets'
    )
    parser.add_argument(
        '--gpt-count',
        type=int,
        default=200,
        help='Number of GPT tickets to generate per rare category (default: 200)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip Kaggle download (use existing data)'
    )
    parser.add_argument(
        '--skip-gpt',
        action='store_true',
        help='Skip GPT augmentation'
    )
    parser.add_argument(
        '--use-existing',
        action='store_true',
        help='Use existing processed data (skip download and processing)'
    )
    
    args = parser.parse_args()
    
    # Load secrets and set up environment
    setup_environment()
    
    print("\n" + "="*100)
    print("  END-TO-END DATA PIPELINE")
    print("="*100)
    
    # Show API key status
    status = check_secrets()
    print(f"\n API Keys Status:")
    print(f"   OpenAI:  {' Configured' if status['openai_ok'] else ' Not configured'}")
    print(f"   Kaggle:  {' Configured' if status['kaggle_ok'] else ' Not configured'}")
    
    print(f"\nConfiguration:")
    print(f"   GPT tickets per category: {args.gpt_count}")
    print(f"   Skip Kaggle download: {args.skip_download}")
    print(f"   Skip GPT augmentation: {args.skip_gpt}")
    print(f"   Use existing processed data: {args.use_existing}")
    
    start_time = time.time()
    
    # Check if using existing processed data
    if args.use_existing:
        print_section("Using Existing Processed Data", "")
        
        required_files = [
            'data/processed/train.csv',
            'data/processed/val.csv',
            'data/processed/test.csv',
            'data/processed/label_mapping.json'
        ]
        
        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            print(" Missing required files:")
            for f in missing:
                print(f"   - {f}")
            print("\n Run without --use-existing to generate fresh data")
            return 1
        
        import pandas as pd
        print(" All required files found:")
        for f in required_files:
            if f.endswith('.csv'):
                df = pd.read_csv(f)
                print(f"    {f}: {len(df):,} tickets")
            else:
                print(f"    {f}")
        
        print_summary(args, start_time)
        return 0
    
    # Step 1: Download IT tickets
    if not args.skip_download:
        if not download_kaggle_data():
            print("\n  Download failed, but continuing with existing data...")
    else:
        print_section("Step 1: Skipped (using existing data)", "")
    
    # Step 2: Process IT tickets
    if not process_it_tickets():
        print("\n Failed to process IT tickets. Exiting.")
        return 1
    
    # Step 3: Generate GPT augmentation (before splitting - we'll add it to train later)
    if not args.skip_gpt:
        if not generate_gpt_augmentation(args.gpt_count):
            print("\n  GPT augmentation failed, but continuing without it...")
    else:
        print_section("Step 3: Skipped (no GPT augmentation)", "")
    
    # Step 4: Split data and add augmentation to train set only
    if not merge_and_preprocess():
        print("\n Failed to split and augment data. Exiting.")
        return 1
    
    # Print summary
    print_summary(args, start_time)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

