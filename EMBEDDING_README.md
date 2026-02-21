# Embedding Pipeline

This pipeline generates embeddings for questions organized by product and category, enabling efficient semantic search and analysis.

## Overview

The pipeline performs the following tasks:

1. Loads question data from the output directory organized by product > category
2. Generates embeddings for questions using a Sentence Transformer model
3. Creates FAISS indices for fast similarity search
4. Stores embeddings and indices by product and category
5. Provides search functionality to find similar questions

## Requirements

- Python 3.7+
- Dependencies from requirements.txt:
  - pandas
  - numpy
  - sentence-transformers
  - faiss-cpu
  - tqdm

## Usage

### 1. Run the Embedding Pipeline

```python
# From the src directory
python embedding.py
```

This will:

- Load the data from the output directory
- Generate embeddings for all questions
- Create FAISS indices for fast similarity search
- Save embeddings, indices, and metadata to the models directory

### 2. Use the API for Embedding and Search

You can also use the API programmatically:

```python
from embedding import EmbeddingPipeline

# Initialize the pipeline
pipeline = EmbeddingPipeline(
    output_dir="output",  # Directory containing organized data
    model_name="all-MiniLM-L6-v2",  # Embedding model to use
    models_dir="models"  # Directory to save embeddings and indices
)

# Generate embeddings (only needed once)
pipeline.create_embeddings()

# Search for similar questions
results = pipeline.search_similar_questions(
    query="How do you handle data encryption?",
    product=None,  # Optional: Limit search to specific product
    category=None,  # Optional: Limit search to specific category
    top_k=5  # Number of results to return
)

# Process search results
for product, category, question, question_id, similarity in results:
    print(f"Product: {product}")
    print(f"Category: {category}")
    print(f"Question: {question}")
    print(f"Question ID: {question_id}")
    print(f"Similarity: {similarity:.4f}")
    print("---")
```

## Output Directory Structure

The pipeline creates the following files in the models directory:

```
models/
  ├── embeddings_metadata.json  # Metadata about all embeddings
  ├── Product1_Category1_embeddings.npy  # Embeddings for questions
  ├── Product1_Category1_index.faiss  # FAISS index for fast search
  ├── ...
```

## Customization

You can customize the pipeline by:

1. **Changing the embedding model**: Pass a different `model_name` to the `EmbeddingPipeline` constructor

   ```python
   pipeline = EmbeddingPipeline(model_name="paraphrase-MiniLM-L6-v2")
   ```

2. **Adjusting search parameters**: Modify the `top_k` parameter in `search_similar_questions`

   ```python
   results = pipeline.search_similar_questions(query="Your question", top_k=10)
   ```

3. **Filtering by product or category**:

   ```python
   # Search only within a specific product
   results = pipeline.search_similar_questions(
       query="Your question",
       product="Danfe_Corp_Product_1"
   )

   # Search only within a specific category
   results = pipeline.search_similar_questions(
       query="Your question",
       category="Data_Security"
   )
   ```

## Performance

The pipeline uses FAISS for fast vector search, which offers significant performance advantages:

- Efficient vector similarity calculations
- Fast retrieval even with millions of vectors
- Low memory consumption with optimized index structures
