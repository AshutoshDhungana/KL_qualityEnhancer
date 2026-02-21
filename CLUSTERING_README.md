# Clustering Pipeline

This pipeline clusters embeddings of questions organized by product and category, identifying similar questions across different products and categories.

## Overview

The clustering pipeline performs the following tasks:

1. Loads question embeddings from the models directory, organized by product and category
2. Reduces the dimensionality of embeddings using UMAP for better clustering performance
3. Clusters the reduced embeddings using HDBSCAN
4. Creates a merged dataset with all questions and their cluster IDs
5. Generates visualizations of the clusters

## Requirements

- Python 3.7+
- Dependencies from requirements.txt:
  - pandas
  - numpy
  - faiss-cpu
  - hdbscan
  - umap-learn
  - matplotlib
  - scikit-learn
  - tqdm

## Usage

### Running the Clustering Pipeline

```bash
# Basic usage
python src/clustering.py

# Specify product to cluster
python src/clustering.py --product Danfe_Corp_Product_1

# Customize clustering parameters
python src/clustering.py --min-cluster-size 10 --min-samples 5

# Customize dimensionality reduction
python src/clustering.py --umap-components 30 --umap-neighbors 20

# Disable visualization
python src/clustering.py --no-visualize
```

### Command Line Arguments

- `--models-dir`: Directory containing embeddings and indices (default: "models")
- `--output-dir`: Directory containing original data (default: "output")
- `--clusters-dir`: Directory to save clustering results (default: "clusters")
- `--product`: Process only specific product (default: all products)
- `--min-cluster-size`: Minimum cluster size for HDBSCAN (default: 5)
- `--min-samples`: Minimum samples parameter for HDBSCAN (default: 2)
- `--umap-components`: Number of components for UMAP dimensionality reduction (default: 50)
- `--umap-neighbors`: Number of neighbors for UMAP (default: 15)
- `--no-visualize`: Disable visualization generation

## Output Files

The pipeline generates the following output files in the clusters directory:

- `all_clusters.csv`: Basic clustering information (cluster_id, product, category, question_id, question)
- `all_merged_clusters.csv`: Complete dataset with all fields from original data plus cluster_id
- `cluster_visualization.png`: Visual representation of clusters using t-SNE

When processing a specific product, files are prefixed with the product name:

- `{product}_clusters.csv`
- `{product}_merged_clusters.csv`

## How Clustering Works

1. **Dimensionality Reduction**: The high-dimensional embeddings are reduced using UMAP to make clustering more effective
2. **Density-Based Clustering**: HDBSCAN identifies clusters based on density regions in the data
3. **Noise Handling**: Points that don't fit well into any cluster are labeled as noise (cluster_id = -1)

## Analysis

The pipeline provides analysis of the clustering results:

- Total number of clusters identified
- Number of noise points
- Top 10 largest clusters
- Distribution of cluster sizes

## Integration with Other Pipelines

This clustering pipeline works after running:

1. Data merging pipeline (`Timed.py`) to combine data from different sources
2. Embedding pipeline (`embedding.py`) to generate embeddings for questions

The complete workflow is:
Data → Merge → Embeddings → Clustering
