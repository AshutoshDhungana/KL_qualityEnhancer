# Enhanced Question Merging Pipeline

This documentation describes the enhanced question merging pipeline that allows merging questions by their IDs and supports prioritization.

## Overview

The enhanced question merging pipeline provides the following features:

1. Merge QA pairs by direct object references or by their IDs
2. Prioritize specific questions during the merge process
3. Use fuzzy matching to compare questions for better merging
4. Use DistilBART for intelligent merging with priority handling

## New API Endpoints

### 1. Merge QA Pairs with Priority

```
POST /merge-qa-pairs
```

**Request Body:**

```json
{
  "pair1": {
    "id": "q1",
    "question": "How do I reset my password?",
    "answer": "To reset your password, click on the 'Forgot Password' link."
  },
  "pair2": {
    "id": "q2",
    "question": "What should I do if I forgot my password?",
    "answer": "If you forgot your password, use the password recovery option."
  },
  "user_id": "user123",
  "similarity_score": 0.85,
  "priority": "pair1" // Optional: "pair1", "pair2", or null
}
```

### 2. Merge QA Pairs by IDs

```
POST /merge-qa-pairs-by-ids
```

**Request Body:**

```json
{
  "cq_ids": ["q1", "q2", "q3"],
  "user_id": "user123",
  "priority_id": "q1", // Optional: ID of the question to prioritize
  "similarity_score": 0.85
}
```

## Prioritization Logic

When a question is prioritized:

1. The prioritized question serves as the base structure
2. Fuzzy matching is used to compare the prioritized question with others
3. DistilBART combines information from other questions into the prioritized question
4. The result maintains the core structure of the prioritized question while incorporating unique information from other questions

## Usage Examples

### Command Line

```bash
# Run example script showing merging with and without priority
python example_merge_script.py
```

### API Usage Example

```python
import requests
import json

# Merge QA pairs with priority
response = requests.post(
    "http://localhost:8000/merge-qa-pairs",
    json={
        "pair1": {
            "id": "q1",
            "question": "How do I reset my password?",
            "answer": "To reset your password, click on the 'Forgot Password' link."
        },
        "pair2": {
            "id": "q2",
            "question": "What should I do if I forgot my password?",
            "answer": "If you forgot your password, use the password recovery option."
        },
        "user_id": "user123",
        "similarity_score": 0.85,
        "priority": "pair1"
    }
)
print(json.dumps(response.json(), indent=2))

# Merge QA pairs by IDs with priority
response = requests.post(
    "http://localhost:8000/merge-qa-pairs-by-ids",
    json={
        "cq_ids": ["q1", "q2", "q3"],
        "user_id": "user123",
        "priority_id": "q1",
        "similarity_score": 0.85
    }
)
print(json.dumps(response.json(), indent=2))
```

## Dependencies

The enhanced pipeline requires the following additional dependencies:

- fuzzywuzzy: For fuzzy string matching
- python-Levenshtein: For improved performance with fuzzy matching
- transformers: For the DistilBART model (when enabled)
- torch: Required for transformers

You can install all required dependencies using:

```bash
pip install -r requirements.txt
```
