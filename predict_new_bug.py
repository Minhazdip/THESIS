import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

print("="*60)
print("BUG RESOLUTION TIME PREDICTOR")
print("="*60)

# Load model
try:
    model = joblib.load('best_model.pkl')
    print("\nâœ“ Model loaded successfully")
except:
    print("\nâœ— Error: Run 'train_models.py' first to create the model")
    exit()

# Example bug for prediction
def predict_bug_resolution(
    priority='Normal',  # Low, Normal, High, Urgent
    summary='',
    description='',
    year=2024,
    month=1,
    day_of_week=0  # 0=Monday, 6=Sunday
):
    """
    Predict resolution time for a new bug
    """
    
    # Priority encoding (based on alphabetical order)
    priority_map = {'High': 0, 'Low': 1, 'Normal': 2, 'Urgent': 3}
    priority_encoded = priority_map.get(priority, 2)
    
    # Text features
    summary_length = len(summary)
    description_length = len(description) if description else 0
    has_description = 1 if description else 0
    keyword_count = len(summary.split())
    
    # Category detection
    text_combined = (summary + ' ' + description).lower()
    
    is_security = int(any(kw in text_combined for kw in ['security', 'cve', 'vulnerability', 'ssl', 'tls', 'encryption']))
    is_performance = int(any(kw in text_combined for kw in ['performance', 'slow', 'latency', 'optimization', 'memory', 'cpu']))
    is_documentation = int(any(kw in text_combined for kw in ['doc', 'documentation', 'readme']))
    is_test = int(any(kw in text_combined for kw in ['test', 'testing', 'junit', 'flaky', 'dtest']))
    is_repair = int(any(kw in text_combined for kw in ['repair', 'anticompaction']))
    is_compaction = int(any(kw in text_combined for kw in ['compaction', 'sstable']))
    is_streaming = int(any(kw in text_combined for kw in ['stream', 'streaming']))
    is_config = int(any(kw in text_combined for kw in ['config', 'configuration', 'yaml']))
    
    # Create feature vector
    features = np.array([[
        priority_encoded, summary_length, description_length, has_description,
        year, month, day_of_week, keyword_count,
        is_security, is_performance, is_documentation, is_test,
        is_repair, is_compaction, is_streaming, is_config
    ]])
    
    # Predict
    prediction = model.predict(features)[0]
    
    return prediction, {
        'priority': priority,
        'summary_length': summary_length,
        'description_length': description_length,
        'categories': {
            'security': bool(is_security),
            'performance': bool(is_performance),
            'documentation': bool(is_documentation),
            'test': bool(is_test),
            'repair': bool(is_repair),
            'compaction': bool(is_compaction),
            'streaming': bool(is_streaming),
            'config': bool(is_config)
        }
    }

# Example predictions
print("\n" + "="*60)
print("EXAMPLE PREDICTIONS")
print("="*60)

examples = [
    {
        'priority': 'Urgent',
        'summary': 'Critical security vulnerability in authentication',
        'description': 'CVE-2024-1234 allows unauthorized access to cluster. Immediate fix required.',
    },
    {
        'priority': 'Normal',
        'summary': 'Update documentation for new feature',
        'description': 'Add examples and usage guide for the new streaming API',
    },
    {
        'priority': 'High',
        'summary': 'Performance degradation in compaction',
        'description': 'Compaction process is slow on large sstables causing memory issues',
    },
    {
        'priority': 'Low',
        'summary': 'Flaky test in repair module',
        'description': 'Test fails intermittently in CI pipeline',
    }
]

for i, example in enumerate(examples, 1):
    print(f"\nExample {i}:")
    print(f"  Priority: {example['priority']}")
    print(f"  Summary: {example['summary']}")
    
    days, info = predict_bug_resolution(**example)
    
    print(f"  â†’ Predicted Resolution Time: {days:.1f} days")
    
    detected_categories = [cat for cat, val in info['categories'].items() if val]
    if detected_categories:
        print(f"  â†’ Detected Categories: {', '.join(detected_categories)}")

print("\n" + "="*60)
print("INTERACTIVE PREDICTION")
print("="*60)

print("\nYou can now predict resolution time for your own bugs!")
print("Enter bug details (or press Enter to skip):\n")

try:
    priority = input("Priority (Low/Normal/High/Urgent) [Normal]: ").strip() or 'Normal'
    summary = input("Summary: ").strip()
    description = input("Description (optional): ").strip()
    
    if summary:
        days, info = predict_bug_resolution(
            priority=priority,
            summary=summary,
            description=description
        )
        
        print("\n" + "-"*60)
        print(f"Predicted Resolution Time: {days:.1f} days ({days/7:.1f} weeks)")
        print("-"*60)
        
        detected_categories = [cat for cat, val in info['categories'].items() if val]
        if detected_categories:
            print(f"Detected Categories: {', '.join(detected_categories)}")
        
        # Provide context
        print("\nContext:")
        if days < 20:
            print("  âœ“ This is expected to be resolved relatively quickly")
        elif days < 50:
            print("  âš  This may take a moderate amount of time")
        else:
            print("  âš  This is expected to take considerable time to resolve")
            
except KeyboardInterrupt:
    print("\n\nPrediction cancelled.")
except Exception as e:
    print(f"\nError: {e}")

print("\nâœ“ Done!")

