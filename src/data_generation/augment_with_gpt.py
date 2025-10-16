"""GPT-based Data Augmentation for Rare Categories"""

import os
import pandas as pd
import json
from openai import OpenAI

RARE_CATEGORIES = [
    "Feature Requests",
    "Integration Issues", 
    "Performance Problems",
    "Security & Compliance"
]

PROMPTS = {
    "Feature Requests": "Generate a realistic customer support ticket requesting a new feature or improvement. Include title and detailed description.",
    "Integration Issues": "Generate a realistic support ticket about integration problems with third-party services or APIs. Include title and detailed description.",
    "Performance Problems": "Generate a realistic support ticket about slow performance, timeouts, or system lag. Include title and detailed description.",
    "Security & Compliance": "Generate a realistic support ticket about security concerns, data privacy, or compliance questions. Include title and detailed description."
}


def generate_ticket(category, client):
    """Generate a single ticket using GPT"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are generating realistic customer support tickets. Return JSON with 'title' and 'description' fields."},
                {"role": "user", "content": PROMPTS[category]}
            ],
            temperature=0.8,
            max_tokens=300
        )
        
        content = response.choices[0].message.content
        
        # Try to parse JSON
        if '{' in content:
            content = content[content.find('{'):content.rfind('}')+1]
            data = json.loads(content)
            return {
                'title': data.get('title', ''),
                'description': data.get('description', ''),
                'category': category
            }
    except Exception as e:
        print(f"   Error: {e}")
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=200, help='Tickets per category')
    parser.add_argument('--yes', action='store_true', help='Skip confirmation')
    args = parser.parse_args()
    
    # Check API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return 1
    
    client = OpenAI(api_key=api_key)
    
    total = len(RARE_CATEGORIES) * args.count
    print(f"🤖 Generating {total} tickets ({args.count} per category)")
    
    if not args.yes:
        confirm = input(f"This will cost ~${total * 0.002:.2f}. Continue? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled")
            return 0
    
    # Generate tickets
    all_tickets = []
    for category in RARE_CATEGORIES:
        print(f"\n📝 Generating {args.count} for {category}...")
        
        for i in range(args.count):
            ticket = generate_ticket(category, client)
            if ticket and ticket['title'] and ticket['description']:
                all_tickets.append(ticket)
                if (i + 1) % 20 == 0:
                    print(f"   Progress: {i+1}/{args.count}")
        
        print(f"   ✅ Generated {len([t for t in all_tickets if t['category'] == category])} tickets")
    
    # Save
    df = pd.DataFrame(all_tickets)
    os.makedirs('data', exist_ok=True)
    output_file = 'data/synthetic_gpt_augmentation.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved {len(df)} tickets to: {output_file}")
    print(f"\n📊 Distribution:")
    for cat, count in df['category'].value_counts().items():
        print(f"   {cat}: {count}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
