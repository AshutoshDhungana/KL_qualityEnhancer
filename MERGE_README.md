# QnA Merge Functionality

This README provides documentation for the question and answer merging functionality in the QnA Content Management Dashboard.

## Overview

The merge functionality allows users to combine similar questions and answers into a single canonical entry. This is particularly useful for:

- Removing duplicate content
- Creating more comprehensive answers
- Standardizing question phrasing
- Maintaining a cleaner knowledge base

## Architecture

The merge functionality consists of three main components:

1. **Backend API**: Provides endpoints for merging QA pairs and saving merged results
2. **Merge Utilities**: Core logic for merging questions and answers using text generation models
3. **Frontend Interface**: A user-friendly review panel for merging similar questions

## Backend API Endpoints

The following API endpoints are available for merging operations:

- `POST /merge-qa-pairs`: Merge two QA pairs using the merge_utils pipeline
- `POST /save-merged-pair`: Save a merged QA pair to a file
- `GET /merged-qa-pairs`: Get a list of all saved merged QA pairs

## Using the Merge Functionality

### Via the Review Panel

1. Navigate to the "Review Panel" in the QnA Content Management Dashboard
2. Use the "Merge Items" tab to compare and merge similar QA pairs
3. Edit the merged result if necessary
4. Save the merged result to create a canonical entry
5. View your merge history in the "Merged History" tab

### Programmatically via API

You can also use the API endpoints directly:

```python
import requests
import json

# API endpoint
API_URL = "http://localhost:8000"

# Example QA pairs to merge
qa_pair1 = {
    "id": "qa1",
    "question": "What are the security measures in place for data access?",
    "answer": "Our system implements role-based access control (RBAC)..."
}

qa_pair2 = {
    "id": "qa2",
    "question": "How does your system control access to data?",
    "answer": "We use role-based access control to manage permissions..."
}

# Merge the QA pairs
merge_request = {
    "pair1": qa_pair1,
    "pair2": qa_pair2,
    "user_id": "admin",
    "similarity_score": 0.85
}

# Call the API
response = requests.post(f"{API_URL}/merge-qa-pairs", json=merge_request)
merged_result = response.json()

# Save the merged result
save_request = {
    "id": merged_result["id"],
    "question": merged_result["question"],
    "answer": merged_result["answer"]
}

save_response = requests.post(f"{API_URL}/save-merged-pair", json=save_request)
```

## Merge Logic

The merge functionality uses the `QuestionMerger` class from `merge_utils.py` to:

1. Analyze similar questions to create a canonical version
2. Combine information from multiple answers
3. Track merge operations for auditing

In a production environment, this would utilize a text generation model like DistilBART for high-quality merges. The current implementation uses a simulated model for demonstration purposes.

## Directory Structure

- `merge_utils.py`: Core utilities for merging questions and answers
- `backend/api.py`: API endpoints for merge operations
- `app.py`: Frontend interface with the Review Panel
- `merged_qa_pairs/`: Directory where merged QA pairs are saved

## Running the System

1. Start the backend API server:

   ```
   cd backend
   python run_api.py
   ```

2. Start the Streamlit frontend:

   ```
   streamlit run app.py
   ```

3. Navigate to the Review Panel to start merging QA pairs
