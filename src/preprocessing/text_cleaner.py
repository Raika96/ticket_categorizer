"""
Text cleaning and PII redaction module
Handles privacy-sensitive data and text normalization
"""

import re
import pandas as pd
from typing import Dict, List, Tuple


class PIIRedactor:
    """Redact Personally Identifiable Information from text"""
    
    # Regex patterns for PII detection
    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        'url': r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)',
        'ticket_id': r'\b(?:ticket|id|case)[\s#:]*[A-Z0-9]{6,}\b',
        'invoice_id': r'\b(?:invoice|inv)[\s#:]*[A-Z0-9]{6,}\b',
        'user_id': r'\b(?:user|uid)[\s#:]*[A-Z0-9]{6,}\b',
        'account_number': r'\b(?:account|acct)[\s#:]*\d{6,}\b',
    }
    
    # Replacement tokens
    REPLACEMENTS = {
        'email': '[EMAIL]',
        'phone': '[PHONE]',
        'ssn': '[SSN]',
        'credit_card': '[CARD]',
        'ip_address': '[IP_ADDRESS]',
        'url': '[URL]',
        'ticket_id': '[TICKET_ID]',
        'invoice_id': '[INVOICE_ID]',
        'user_id': '[USER_ID]',
        'account_number': '[ACCOUNT_NUMBER]',
    }
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.stats = {key: 0 for key in self.PATTERNS.keys()}
    
    def redact(self, text: str) -> str:
        """
        Redact all PII from text
        
        Args:
            text: Input text string
            
        Returns:
            Text with PII redacted
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        redacted_text = text
        
        # Apply each pattern
        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, redacted_text, re.IGNORECASE)
            if matches:
                self.stats[pii_type] += len(matches)
                redacted_text = re.sub(pattern, self.REPLACEMENTS[pii_type], 
                                      redacted_text, flags=re.IGNORECASE)
        
        return redacted_text
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics on redacted PII"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {key: 0 for key in self.PATTERNS.keys()}


class TextNormalizer:
    """Normalize and clean text data"""
    
    def __init__(self):
        # Common text artifacts and their replacements
        self.artifact_patterns = [
            (r'\r\n|\r|\n', ' '),  # Line breaks to spaces
            (r'\t', ' '),  # Tabs to spaces
            (r'&nbsp;', ' '),  # HTML non-breaking space
            (r'&amp;', '&'),  # HTML ampersand
            (r'&lt;', '<'),  # HTML less than
            (r'&gt;', '>'),  # HTML greater than
            (r'&quot;', '"'),  # HTML quote
            (r'&#\d+;', ''),  # HTML entities
            (r'<[^>]+>', ''),  # HTML tags
        ]
        
        # Patterns for cleaning
        self.noise_patterns = [
            (r'\.{3,}', '...'),  # Multiple dots to ellipsis
            (r'!{2,}', '!'),  # Multiple exclamation marks
            (r'\?{2,}', '?'),  # Multiple question marks
            (r'-{3,}', ' '),  # Multiple dashes
            (r'={3,}', ' '),  # Multiple equals
            (r'_{3,}', ' '),  # Multiple underscores
        ]
    
    def normalize(self, text: str) -> str:
        """
        Normalize text by cleaning artifacts and standardizing format
        
        Args:
            text: Input text string
            
        Returns:
            Normalized text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        normalized = text
        
        # Remove HTML artifacts
        for pattern, replacement in self.artifact_patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        # Clean noise patterns
        for pattern, replacement in self.noise_patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Remove leading/trailing whitespace
        normalized = normalized.strip()
        
        return normalized
    
    def remove_non_printable(self, text: str) -> str:
        """Remove non-printable characters"""
        if pd.isna(text):
            return ""
        return ''.join(char for char in text if char.isprintable() or char.isspace())
    
    def normalize_case(self, text: str) -> str:
        """Convert text to lowercase"""
        if pd.isna(text):
            return ""
        return text.lower()


class TextCleaner:
    """Comprehensive text cleaning pipeline"""
    
    def __init__(self, 
                 min_length: int = 10,
                 max_length: int = 2000,
                 redact_pii: bool = True,
                 verbose: bool = False):
        """
        Initialize text cleaner
        
        Args:
            min_length: Minimum acceptable text length
            max_length: Maximum text length (will truncate)
            redact_pii: Whether to redact PII
            verbose: Print detailed statistics
        """
        self.min_length = min_length
        self.max_length = max_length
        self.redact_pii_flag = redact_pii
        self.verbose = verbose
        
        self.pii_redactor = PIIRedactor(verbose=verbose)
        self.normalizer = TextNormalizer()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'too_short': 0,
            'too_long': 0,
            'truncated': 0,
            'empty_after_cleaning': 0,
            'pii_redacted': {}
        }
    
    def clean_text(self, text: str) -> str:
        """
        Apply full cleaning pipeline to text
        
        Pipeline:
        1. Remove non-printable characters
        2. Redact PII (if enabled)
        3. Normalize text (HTML artifacts, case, whitespace)
        4. Handle length constraints
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        self.stats['total_processed'] += 1
        
        # Step 1: Remove non-printable characters
        cleaned = self.normalizer.remove_non_printable(text)
        
        # Step 2: Redact PII
        if self.redact_pii_flag:
            cleaned = self.pii_redactor.redact(cleaned)
        
        # Step 3: Normalize
        cleaned = self.normalizer.normalize(cleaned)
        
        # Step 4: Handle length
        if len(cleaned) < self.min_length:
            self.stats['too_short'] += 1
            return ""  # Will be filtered out later
        
        if len(cleaned) > self.max_length:
            self.stats['too_long'] += 1
            self.stats['truncated'] += 1
            cleaned = cleaned[:self.max_length]
        
        if not cleaned.strip():
            self.stats['empty_after_cleaning'] += 1
            return ""
        
        return cleaned
    
    def clean_dataframe(self, df: pd.DataFrame, 
                       text_columns: List[str]) -> pd.DataFrame:
        """
        Clean text in a DataFrame
        
        Args:
            df: Input DataFrame
            text_columns: List of column names containing text
            
        Returns:
            DataFrame with cleaned text
        """
        df = df.copy()
        
        print(f"\n Cleaning text in columns: {text_columns}")
        print(f"   Min length: {self.min_length} chars")
        print(f"   Max length: {self.max_length} chars")
        print(f"   PII redaction: {'Enabled' if self.redact_pii_flag else 'Disabled'}")
        
        # Reset stats
        self.stats = {
            'total_processed': 0,
            'too_short': 0,
            'too_long': 0,
            'truncated': 0,
            'empty_after_cleaning': 0,
        }
        self.pii_redactor.reset_stats()
        
        # Clean each text column
        for col in text_columns:
            if col in df.columns:
                print(f"   Processing '{col}'...")
                df[f'{col}_clean'] = df[col].apply(self.clean_text)
        
        # Get PII stats
        self.stats['pii_redacted'] = self.pii_redactor.get_stats()
        
        # Print statistics
        if self.verbose:
            self.print_stats()
        
        return df
    
    def print_stats(self):
        """Print cleaning statistics"""
        print(f"\nCleaning Statistics:")
        print(f"   Total processed: {self.stats['total_processed']:,}")
        print(f"   Too short (< {self.min_length}): {self.stats['too_short']:,}")
        print(f"   Too long (> {self.max_length}): {self.stats['too_long']:,}")
        print(f"   Truncated: {self.stats['truncated']:,}")
        print(f"   Empty after cleaning: {self.stats['empty_after_cleaning']:,}")
        
        if self.redact_pii_flag:
            print(f"\nPII Redaction Statistics:")
            for pii_type, count in self.stats['pii_redacted'].items():
                if count > 0:
                    print(f"   {pii_type}: {count:,} instances")
            
            total_pii = sum(self.stats['pii_redacted'].values())
            if total_pii > 0:
                print(f"   Total PII redacted: {total_pii:,} instances")
            else:
                print(f"   No PII detected")
    
    def get_stats(self) -> Dict:
        """Get cleaning statistics"""
        return self.stats.copy()


def handle_duplicates(df: pd.DataFrame, 
                     text_cols: List[str],
                     keep: str = 'first') -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate tickets
    
    Args:
        df: Input DataFrame
        text_cols: Columns to check for duplicates
        keep: Which duplicate to keep ('first', 'last', False)
        
    Returns:
        (cleaned DataFrame, number of duplicates removed)
    """
    initial_count = len(df)
    
    # Check for exact duplicates
    df_dedup = df.drop_duplicates(subset=text_cols, keep=keep)
    
    duplicates_removed = initial_count - len(df_dedup)
    
    if duplicates_removed > 0:
        print(f"\nDuplicate Handling:")
        print(f"   Found {duplicates_removed:,} duplicate tickets")
        print(f"   Removed duplicates, keeping '{keep}' occurrence")
        print(f"   Remaining tickets: {len(df_dedup):,}")
    else:
        print(f"\nNo duplicate tickets found")
    
    return df_dedup, duplicates_removed


def handle_extreme_cases(df: pd.DataFrame,
                        length_col: str,
                        min_percentile: float = 1,
                        max_percentile: float = 99) -> Tuple[pd.DataFrame, Dict]:
    """
    Handle extreme cases (very short or very long tickets)
    
    Args:
        df: Input DataFrame
        length_col: Column containing text length
        min_percentile: Lower percentile threshold
        max_percentile: Upper percentile threshold
        
    Returns:
        (filtered DataFrame, statistics dict)
    """
    initial_count = len(df)
    
    # Calculate percentiles
    min_threshold = df[length_col].quantile(min_percentile / 100)
    max_threshold = df[length_col].quantile(max_percentile / 100)
    
    # Filter
    df_filtered = df[
        (df[length_col] >= min_threshold) & 
        (df[length_col] <= max_threshold)
    ].copy()
    
    removed = initial_count - len(df_filtered)
    
    stats = {
        'initial_count': initial_count,
        'final_count': len(df_filtered),
        'removed': removed,
        'min_threshold': min_threshold,
        'max_threshold': max_threshold,
        'too_short': len(df[df[length_col] < min_threshold]),
        'too_long': len(df[df[length_col] > max_threshold])
    }
    
    if removed > 0:
        print(f"\nWarning: Extreme Case Handling:")
        print(f"   Length thresholds: {min_threshold:.0f} - {max_threshold:.0f} chars")
        print(f"   Too short (< {min_percentile}th percentile): {stats['too_short']:,}")
        print(f"   Too long (> {max_percentile}th percentile): {stats['too_long']:,}")
        print(f"   Total removed: {removed:,}")
        print(f"   Remaining tickets: {len(df_filtered):,}")
    
    return df_filtered, stats


if __name__ == "__main__":
    # Test the cleaning functions
    test_text = """
    Hello! My email is john.doe@example.com and phone is 555-123-4567.
    Please contact me at https://example.com/support
    My account number is ACCT 123456789.
    
    This has    multiple    spaces and
    line breaks...
    
    Thanks!!!
    """
    
    print("="*70)
    print(" Testing Text Cleaning & PII Redaction")
    print("="*70)
    
    cleaner = TextCleaner(redact_pii=True, verbose=True)
    cleaned = cleaner.clean_text(test_text)
    
    print(f"\nOriginal:\n{test_text}")
    print(f"\nCleaned:\n{cleaned}")
    
    cleaner.print_stats()

