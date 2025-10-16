#!/usr/bin/env python3
"""
Helper module to load API keys and secrets from secrets.json

Usage:
    from load_secrets import load_secrets
    
    secrets = load_secrets()
    openai_key = secrets['openai']['api_key']
"""

import os
import json
from pathlib import Path


def load_secrets(secrets_file='secrets.json'):
    """
    Load secrets from secrets.json file
    
    Priority order:
    1. secrets.json file (if exists)
    2. Environment variables
    3. ~/.kaggle/kaggle.json for Kaggle credentials
    
    Returns:
        dict: Dictionary with 'openai' and 'kaggle' keys
    """
    secrets = {
        'openai': {
            'api_key': None
        },
        'kaggle': {
            'username': None,
            'key': None
        }
    }
    
    # Try to load from secrets.json
    if os.path.exists(secrets_file):
        try:
            with open(secrets_file, 'r') as f:
                file_secrets = json.load(f)
                
            # Load OpenAI key
            if 'openai' in file_secrets and 'api_key' in file_secrets['openai']:
                secrets['openai']['api_key'] = file_secrets['openai']['api_key']
            
            # Load Kaggle credentials
            if 'kaggle' in file_secrets:
                if 'username' in file_secrets['kaggle']:
                    secrets['kaggle']['username'] = file_secrets['kaggle']['username']
                if 'key' in file_secrets['kaggle']:
                    secrets['kaggle']['key'] = file_secrets['kaggle']['key']
                    
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {secrets_file}: {e}")
    
    # Fallback to environment variables
    if not secrets['openai']['api_key']:
        secrets['openai']['api_key'] = os.environ.get('OPENAI_API_KEY')
    
    if not secrets['kaggle']['username']:
        secrets['kaggle']['username'] = os.environ.get('KAGGLE_USERNAME')
    
    if not secrets['kaggle']['key']:
        secrets['kaggle']['key'] = os.environ.get('KAGGLE_KEY')
    
    # Fallback to ~/.kaggle/kaggle.json for Kaggle
    if not (secrets['kaggle']['username'] and secrets['kaggle']['key']):
        kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
        if kaggle_json.exists():
            try:
                with open(kaggle_json, 'r') as f:
                    kaggle_creds = json.load(f)
                secrets['kaggle']['username'] = kaggle_creds.get('username')
                secrets['kaggle']['key'] = kaggle_creds.get('key')
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse {kaggle_json}: {e}")
    
    return secrets


def setup_environment(secrets=None):
    """
    Set up environment variables from secrets
    
    Args:
        secrets: Dictionary from load_secrets(), or None to load automatically
    """
    if secrets is None:
        secrets = load_secrets()
    
    # Set OpenAI API key
    if secrets['openai']['api_key'] and secrets['openai']['api_key'] != 'your-openai-api-key-here':
        os.environ['OPENAI_API_KEY'] = secrets['openai']['api_key']
    
    # Set Kaggle credentials
    if secrets['kaggle']['username'] and secrets['kaggle']['username'] != 'your-kaggle-username':
        os.environ['KAGGLE_USERNAME'] = secrets['kaggle']['username']
    
    if secrets['kaggle']['key'] and secrets['kaggle']['key'] != 'your-kaggle-key':
        os.environ['KAGGLE_KEY'] = secrets['kaggle']['key']


def check_secrets():
    """
    Check if required secrets are configured
    
    Returns:
        dict: Dictionary with 'openai_ok' and 'kaggle_ok' boolean flags
    """
    secrets = load_secrets()
    
    status = {
        'openai_ok': False,
        'kaggle_ok': False,
        'openai_key': None,
        'kaggle_username': None
    }
    
    # Check OpenAI
    api_key = secrets['openai']['api_key']
    if api_key and api_key != 'your-openai-api-key-here':
        status['openai_ok'] = True
        status['openai_key'] = api_key[:20] + '...' if len(api_key) > 20 else api_key
    
    # Check Kaggle
    username = secrets['kaggle']['username']
    key = secrets['kaggle']['key']
    if username and username != 'your-kaggle-username' and key and key != 'your-kaggle-key':
        status['kaggle_ok'] = True
        status['kaggle_username'] = username
    
    return status


if __name__ == '__main__':
    """Test the secrets loading"""
    print("Checking API Keys Configuration\n")
    
    status = check_secrets()
    
    print("OpenAI API:")
    if status['openai_ok']:
        print(f"  Configured: {status['openai_key']}")
    else:
        print("  Not configured")
        print("     Set in secrets.json or export OPENAI_API_KEY")
    
    print("\nKaggle API:")
    if status['kaggle_ok']:
        print(f"  Configured: {status['kaggle_username']}")
    else:
        print("  Not configured")
        print("     Set in secrets.json or ~/.kaggle/kaggle.json")
    
    print("\n" + "="*70)
    if status['openai_ok'] and status['kaggle_ok']:
        print("All API keys configured!")
    else:
        print("Warning: Some API keys are missing. See secrets.json.example")
    print("="*70)

