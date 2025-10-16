"""
Specialized processor for IT Service Ticket Classification Dataset from Kaggle.
Dataset: https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset

This dataset is ideal for our SaaS support ticket classification task.
"""

import pandas as pd
import os
import json
from datetime import datetime
from tqdm import tqdm
import random
import numpy as np

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)


class ITServiceTicketProcessor:
    """Process IT Service Ticket Classification Dataset"""
    
    # Our target categories
    TARGET_CATEGORIES = {
        0: "Billing & Payments",
        1: "Account Access / Login Issues",
        2: "Bug / Defect Reports",
        3: "Feature Requests",
        4: "Integration Issues",
        5: "Performance Problems",
        6: "Security & Compliance",
        7: "How-To / Product Usage Questions",
        8: "General Inquiries / Other"
    }
    
    # Mapping from common IT ticket categories to our categories
    CATEGORY_MAPPING = {
        # Account/Access related
        'account': 'Account Access / Login Issues',
        'login': 'Account Access / Login Issues',
        'password': 'Account Access / Login Issues',
        'access': 'Account Access / Login Issues',
        'authentication': 'Account Access / Login Issues',
        'credentials': 'Account Access / Login Issues',
        'sso': 'Account Access / Login Issues',
        'mfa': 'Account Access / Login Issues',
        
        # Bug/Issue related
        'incident': 'Bug / Defect Reports',
        'bug': 'Bug / Defect Reports',
        'error': 'Bug / Defect Reports',
        'defect': 'Bug / Defect Reports',
        'issue': 'Bug / Defect Reports',
        'problem': 'Bug / Defect Reports',
        'failure': 'Bug / Defect Reports',
        'broken': 'Bug / Defect Reports',
        
        # Request related
        'service request': 'How-To / Product Usage Questions',
        'request': 'How-To / Product Usage Questions',
        'how to': 'How-To / Product Usage Questions',
        'question': 'How-To / Product Usage Questions',
        'help': 'How-To / Product Usage Questions',
        'inquiry': 'How-To / Product Usage Questions',
        'information': 'How-To / Product Usage Questions',
        
        # Enhancement/Feature
        'enhancement': 'Feature Requests',
        'feature': 'Feature Requests',
        'improvement': 'Feature Requests',
        'change request': 'Feature Requests',
        
        # Performance
        'performance': 'Performance Problems',
        'slow': 'Performance Problems',
        'timeout': 'Performance Problems',
        'latency': 'Performance Problems',
        'degradation': 'Performance Problems',
        
        # Integration
        'integration': 'Integration Issues',
        'api': 'Integration Issues',
        'webhook': 'Integration Issues',
        'sync': 'Integration Issues',
        'connection': 'Integration Issues',
        
        # Security
        'security': 'Security & Compliance',
        'compliance': 'Security & Compliance',
        'audit': 'Security & Compliance',
        'privacy': 'Security & Compliance',
        'gdpr': 'Security & Compliance',
        'hipaa': 'Security & Compliance',
        
        # Billing
        'billing': 'Billing & Payments',
        'payment': 'Billing & Payments',
        'invoice': 'Billing & Payments',
        'subscription': 'Billing & Payments',
        'charge': 'Billing & Payments',
        'refund': 'Billing & Payments',
    }
    
    # Keywords for intelligent classification
    CLASSIFICATION_KEYWORDS = {
        "Billing & Payments": [
            'billing', 'payment', 'invoice', 'refund', 'charge', 'subscription',
            'price', 'pricing', 'cost', 'fee', 'card', 'transaction', 'pay'
        ],
        "Account Access / Login Issues": [
            'login', 'password', 'access', 'authentication', 'mfa', '2fa',
            'sso', 'locked', 'lock', 'username', 'sign in', 'credentials'
        ],
        "Bug / Defect Reports": [
            'bug', 'error', 'broken', 'crash', 'fail', 'not working',
            'issue', 'problem', 'defect', 'wrong', 'incorrect', 'glitch'
        ],
        "Feature Requests": [
            'feature', 'enhancement', 'improvement', 'suggestion', 'request',
            'would like', 'please add', 'missing', 'need'
        ],
        "Integration Issues": [
            'api', 'integration', 'webhook', 'sync', 'connect', 'third party',
            'external', 'endpoint', 'oauth', 'rest'
        ],
        "Performance Problems": [
            'slow', 'performance', 'timeout', 'latency', 'lag', 'loading',
            'freeze', 'hang', 'speed', 'delay'
        ],
        "Security & Compliance": [
            'security', 'privacy', 'gdpr', 'hipaa', 'compliance', 'encrypt',
            'breach', 'vulnerability', 'audit', 'unauthorized'
        ],
        "How-To / Product Usage Questions": [
            'how to', 'how do', 'help', 'guide', 'tutorial', 'setup',
            'configure', 'use', 'using', 'question', 'explain'
        ]
    }
    
    def classify_by_keywords(self, text):
        """Classify ticket using keyword matching"""
        if not text or pd.isna(text):
            return "General Inquiries / Other"
        
        text_lower = str(text).lower()
        scores = {category: 0 for category in self.TARGET_CATEGORIES.values()}
        
        # Score each category based on keyword matches
        for category, keywords in self.CLASSIFICATION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Multi-word keywords get more weight
                    weight = len(keyword.split())
                    scores[category] += weight
        
        # Return category with highest score
        max_score = max(scores.values())
        if max_score == 0:
            return "General Inquiries / Other"
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def map_category(self, original_category, text=""):
        """
        Map original dataset category to our target categories
        
        Args:
            original_category: Category from the original dataset
            text: Ticket text for fallback classification
        """
        if pd.isna(original_category):
            return self.classify_by_keywords(text)
        
        category_lower = str(original_category).lower()
        
        # Try direct mapping first
        for key, target_category in self.CATEGORY_MAPPING.items():
            if key in category_lower:
                return target_category
        
        # Fallback to keyword-based classification
        return self.classify_by_keywords(text)
    
    def process_dataset(self, filepath, max_samples=None):
        """
        Process IT Service Ticket Classification Dataset
        
        Args:
            filepath: Path to the CSV file
            max_samples: Maximum number of samples to process (None = all)
        
        Returns:
            DataFrame in our standard format
        """
        print(f"\n{'='*70}")
        print(" Processing IT Service Ticket Classification Dataset")
        print(f"{'='*70}")
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            print(f"\nTo download:")
            print("1. Visit: https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset")
            print("2. Download the dataset")
            print("3. Extract to: data/external/")
            print("\nOr use Kaggle CLI:")
            print("kaggle datasets download -d adisongoh/it-service-ticket-classification-dataset")
            print("unzip it-service-ticket-classification-dataset.zip -d data/external/")
            return pd.DataFrame()
        
        print(f"📁 Loading from: {filepath}")
        
        # Load dataset
        if max_samples:
            df = pd.read_csv(filepath, nrows=max_samples)
        else:
            df = pd.read_csv(filepath)
        
        print(f"✓ Loaded {len(df)} rows")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Display first few rows to understand structure
        print(f"\n📊 Sample data:")
        print(df.head(2))
        
        # Auto-detect columns
        title_col = self._detect_column(df, ['title', 'subject', 'summary', 'short_description', 'ticket_title'])
        desc_col = self._detect_column(df, ['description', 'details', 'body', 'text', 'long_description', 'ticket_description', 'document'])
        category_col = self._detect_column(df, ['category', 'type', 'ticket_type', 'classification', 'class', 'topic_group'])
        priority_col = self._detect_column(df, ['priority', 'severity', 'urgency'])
        
        print(f"\n🔍 Detected columns:")
        print(f"   Title: {title_col}")
        print(f"   Description: {desc_col}")
        print(f"   Category: {category_col}")
        print(f"   Priority: {priority_col}")
        
        if category_col:
            print(f"\n📊 Original categories in dataset:")
            print(df[category_col].value_counts())
        
        # Process tickets
        tickets = []
        print(f"\n⚙️  Processing tickets...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            title = str(row[title_col]) if title_col and not pd.isna(row[title_col]) else ""
            description = str(row[desc_col]) if desc_col and not pd.isna(row[desc_col]) else ""
            original_category = row[category_col] if category_col else None
            
            # Skip if both title and description are empty
            if not title.strip() and not description.strip():
                continue
            
            # Combine for classification
            combined_text = f"{title} {description}"
            
            # Map category
            mapped_category = self.map_category(original_category, combined_text)
            
            # Get priority
            if priority_col and not pd.isna(row[priority_col]):
                priority = str(row[priority_col])
                # Normalize priority
                priority_lower = priority.lower()
                if priority_lower in ['high', 'critical', 'urgent', '1']:
                    priority = 'High'
                elif priority_lower in ['medium', 'normal', '2']:
                    priority = 'Medium'
                elif priority_lower in ['low', '3']:
                    priority = 'Low'
                else:
                    priority = 'Medium'
            else:
                priority = 'Medium'
            
            # Create ticket
            ticket = {
                'ticket_id': f"it_service_{idx}",
                'title': title if title else description[:100],
                'description': description if description else title,
                'category': mapped_category,
                'priority': priority,
                'created_at': datetime.now().isoformat(),
                'customer_email': 'customer@company.com',
                'customer_name': 'IT User',
                'source': 'kaggle_it_service',
                'original_category': str(original_category) if original_category else None
            }
            tickets.append(ticket)
        
        result_df = pd.DataFrame(tickets)
        
        print(f"\n✅ Processed {len(result_df)} tickets")
        
        if len(result_df) > 0 and 'category' in result_df.columns:
            print(f"\n📊 Category distribution (mapped to our 9 categories):")
            print(result_df['category'].value_counts())
        else:
            print(f"\n⚠️  Warning: No tickets processed or 'category' column missing")
            if len(result_df) > 0:
                print(f"   Available columns: {list(result_df.columns)}")
        
        if 'original_category' in result_df.columns:
            print(f"\n🔄 Category mapping summary:")
            mapping_df = result_df.groupby(['original_category', 'category']).size().reset_index(name='count')
            for _, row in mapping_df.iterrows():
                print(f"   {row['original_category']} → {row['category']} ({row['count']} tickets)")
        
        return result_df
    
    def _detect_column(self, df, possible_names):
        """Auto-detect column by searching for possible names"""
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in possible_names):
                return col
        return None


def generate_dataset_from_it_service(output_dir='data/raw', max_samples=None):
    """
    Main function to generate dataset from IT Service Ticket dataset
    
    Args:
        output_dir: Directory to save processed data
        max_samples: Maximum samples to process (None = all)
    """
    print("="*70)
    print(" IT SERVICE TICKET DATASET PIPELINE")
    print("="*70)
    
    processor = ITServiceTicketProcessor()
    
    # Try common filenames
    possible_files = [
        'data/external/it_service_tickets.csv',
        'data/external/IT_Service_Tickets.csv',
        'data/external/tickets.csv',
        'data/external/train.csv',
        'data/external/data.csv'
    ]
    
    # Find the file
    filepath = None
    for file in possible_files:
        if os.path.exists(file):
            filepath = file
            break
    
    # If not found, check all CSV files in external directory
    if not filepath and os.path.exists('data/external'):
        csv_files = [f for f in os.listdir('data/external') if f.endswith('.csv')]
        if csv_files:
            filepath = os.path.join('data/external', csv_files[0])
            print(f"⚠️  Using first CSV file found: {filepath}")
    
    if not filepath:
        print("❌ No dataset file found!")
        print("\n📥 To download the IT Service Ticket dataset:")
        print("="*70)
        print("\nOption 1: Kaggle CLI")
        print("  kaggle datasets download -d adisongoh/it-service-ticket-classification-dataset")
        print("  unzip it-service-ticket-classification-dataset.zip -d data/external/")
        print("\nOption 2: Manual Download")
        print("  1. Visit: https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset")
        print("  2. Click 'Download'")
        print("  3. Extract CSV to data/external/ directory")
        print("="*70)
        return None
    
    # Process the dataset
    df = processor.process_dataset(filepath, max_samples=max_samples)
    
    if df.empty:
        return None
    
    # Save to the expected location for the pipeline
    output_path = 'data/processed_it_tickets.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Dataset saved to: {output_path}")
    
    # Also save to the raw directory for reference
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, 'support_tickets_it_service.csv')
    df.to_csv(raw_path, index=False)
    print(f"✅ Copy saved to: {raw_path}")
    
    # Statistics
    print(f"\n{'='*70}")
    print(" DATASET STATISTICS")
    print(f"{'='*70}")
    print(f"\nTotal tickets: {len(df)}")
    print(f"\nCategory distribution:")
    for category, count in df['category'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"  {category:<40} {count:>5} ({percentage:>5.1f}%)")
    
    print(f"\nPriority distribution:")
    print(df['priority'].value_counts())
    
    # Text statistics
    df['title_length'] = df['title'].str.len()
    df['description_length'] = df['description'].str.len()
    
    print(f"\nText statistics:")
    print(f"  Title - Mean: {df['title_length'].mean():.0f}, Min: {df['title_length'].min()}, Max: {df['title_length'].max()}")
    print(f"  Description - Mean: {df['description_length'].mean():.0f}, Min: {df['description_length'].min()}, Max: {df['description_length'].max()}")
    
    # Save metadata
    metadata = {
        'dataset_name': 'IT Service Ticket Classification',
        'source': 'Kaggle - adisongoh',
        'total_tickets': len(df),
        'generation_date': datetime.now().isoformat(),
        'category_distribution': df['category'].value_counts().to_dict(),
        'priority_distribution': df['priority'].value_counts().to_dict(),
        'statistics': {
            'avg_title_length': float(df['title_length'].mean()),
            'avg_description_length': float(df['description_length'].mean()),
        }
    }
    
    metadata_path = os.path.join(output_dir, 'it_service_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to: {metadata_path}")
    
    print(f"\n{'='*70}")
    print(" SUCCESS! Ready for model training")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("1. Explore: jupyter notebook notebooks/01_eda.ipynb")
    print("2. Preprocess: python src/preprocessing/clean_data.py")
    print("3. Train: python src/models/train.py")
    
    return df


if __name__ == "__main__":
    df = generate_dataset_from_it_service(max_samples=None)

