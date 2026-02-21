import os
import pandas as pd
import numpy as np
import json
import faiss
import hdbscan
import logging
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("clustering_pipeline.log"), logging.StreamHandler()]
)
logger = logging.getLogger("clustering_pipeline")

def normalize_path(path):
    """Helper function to normalize and make path absolute"""
    if path.startswith(".."):
        # Handle relative paths like ../output
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_dir, path))
    elif os.path.isabs(path):
        return path
    else:
        # Handle relative paths without ../
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        return os.path.join(base_dir, path)

class ClusteringPipeline:
    def __init__(self, 
                 models_dir="models",
                 output_dir="output",
                 clusters_dir="clusters",
                 min_cluster_size=5,
                 min_samples=2,
                 umap_components=50,
                 umap_neighbors=15,
                 visualize=True):
        """
        Initialize the clustering pipeline
        
        Args:
            models_dir: Directory containing embeddings and indices
            output_dir: Directory containing original data organized by product > category
            clusters_dir: Directory to save clustering results
            min_cluster_size: Minimum size of clusters for HDBSCAN
            min_samples: Minimum samples parameter for HDBSCAN
            umap_components: Number of components for UMAP dimensionality reduction
            umap_neighbors: Number of neighbors for UMAP
            visualize: Whether to generate visualization plots
        """
        self.models_dir = normalize_path(models_dir)
        self.output_dir = normalize_path(output_dir)
        self.clusters_dir = normalize_path(clusters_dir)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.umap_components = umap_components
        self.umap_neighbors = umap_neighbors
        self.visualize = visualize
        
        # Create clusters directory if it doesn't exist
        Path(self.clusters_dir).mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        self.metadata_path = os.path.join(self.models_dir, "embeddings_metadata.json")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        logger.info(f"Initialized clustering pipeline")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Clusters directory: {self.clusters_dir}")
    
    def load_embeddings(self, product=None):
        """
        Load embeddings for specific product or all products
        
        Args:
            product: Product to load embeddings for. If None, load all products.
            
        Returns:
            Dictionary with embedding data
        """
        embedding_data = []
        products = [product] if product else list(self.metadata["products"].keys())
        
        for p in products:
            logger.info(f"Loading embeddings for product: {p}")
            
            product_dir = os.path.join(self.models_dir, p)
            if not os.path.exists(product_dir):
                logger.warning(f"Product directory not found: {product_dir}")
                continue
                
            categories = self.metadata["products"][p]["categories"].keys()
            
            for c in tqdm(categories, desc=f"Loading categories for {p}"):
                category_data = self.metadata["products"][p]["categories"][c]
                
                # Load embeddings
                embeddings_path = os.path.join(product_dir, category_data["embeddings_file"])
                
                if not os.path.exists(embeddings_path):
                    logger.warning(f"Embeddings file not found: {embeddings_path}")
                    continue
                
                embeddings = np.load(embeddings_path)
                question_ids = category_data["question_ids"]
                questions = category_data["questions"]
                
                # Add embeddings to the list
                for i, (emb, q_id, q) in enumerate(zip(embeddings, question_ids, questions)):
                    embedding_data.append({
                        "product": p,
                        "category": c,
                        "question_id": q_id,
                        "question": q,
                        "embedding": emb
                    })
        
        logger.info(f"Loaded {len(embedding_data)} embeddings")
        return embedding_data
    
    def reduce_dimensions(self, embeddings, method='umap'):
        """
        Reduce dimensions of embeddings for better clustering
        
        Args:
            embeddings: List of embeddings
            method: Dimensionality reduction method ('umap' or 'tsne')
            
        Returns:
            Reduced embeddings
        """
        if len(embeddings) == 0:
            logger.warning("No embeddings to reduce dimensions")
            return []
        
        # Convert list of embeddings to numpy array
        X = np.array([e for e in embeddings])
        
        logger.info(f"Reducing dimensions from {X.shape[1]} to {self.umap_components} using {method}")
        
        if method == 'umap':
            # UMAP for dimensionality reduction
            start_time = time.time()
            reducer = umap.UMAP(n_components=self.umap_components, n_neighbors=self.umap_neighbors, min_dist=0.0)
            reduced_embeddings = reducer.fit_transform(X)
            logger.info(f"UMAP completed in {time.time() - start_time:.2f} seconds")
        elif method == 'tsne':
            # t-SNE for dimensionality reduction
            start_time = time.time()
            tsne = TSNE(n_components=self.umap_components, perplexity=30, n_iter=1000)
            reduced_embeddings = tsne.fit_transform(X)
            logger.info(f"t-SNE completed in {time.time() - start_time:.2f} seconds")
        else:
            logger.warning(f"Unknown dimensionality reduction method: {method}")
            reduced_embeddings = X
        
        return reduced_embeddings
    
    def cluster_embeddings(self, reduced_embeddings, embedding_data):
        """
        Cluster embeddings using HDBSCAN
        
        Args:
            reduced_embeddings: Reduced-dimension embeddings
            embedding_data: Original embedding data
            
        Returns:
            Dictionary with clustered data
        """
        if len(reduced_embeddings) == 0:
            logger.warning("No embeddings to cluster")
            return []
        
        # Cluster embeddings
        logger.info(f"Clustering {len(reduced_embeddings)} embeddings with HDBSCAN")
        logger.info(f"Parameters: min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples}")
        
        start_time = time.time()
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean'
        )
        cluster_labels = clusterer.fit_predict(reduced_embeddings)
        logger.info(f"Clustering completed in {time.time() - start_time:.2f} seconds")
        
        # Count clusters
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        logger.info(f"Number of clusters: {n_clusters}")
        logger.info(f"Number of noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.2f}%)")
        
        # Add cluster labels to embedding data
        for i, label in enumerate(cluster_labels):
            embedding_data[i]["cluster_id"] = int(label)
        
        # Create a dictionary of clusters
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[int(label)].append(embedding_data[i])
        
        return clusters, reduced_embeddings, cluster_labels
    
    def visualize_clusters(self, reduced_embeddings, cluster_labels, output_file):
        """
        Visualize clusters using t-SNE and save the plot
        
        Args:
            reduced_embeddings: Reduced-dimension embeddings
            cluster_labels: Cluster labels
            output_file: File to save the plot to
        """
        if not self.visualize:
            return
        
        logger.info("Generating visualization of clusters")
        
        # Further reduce dimensions to 2D for visualization
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
        vis_embeddings = tsne.fit_transform(reduced_embeddings)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Plot noise points as gray
        noise_mask = cluster_labels == -1
        plt.scatter(
            vis_embeddings[noise_mask, 0],
            vis_embeddings[noise_mask, 1],
            c='gray',
            marker='.',
            alpha=0.5,
            label='Noise'
        )
        
        # Plot clusters
        for cluster in set(cluster_labels):
            if cluster == -1:
                continue
            
            mask = cluster_labels == cluster
            plt.scatter(
                vis_embeddings[mask, 0],
                vis_embeddings[mask, 1],
                marker='o',
                alpha=0.8,
                label=f'Cluster {cluster}'
            )
        
        plt.title('Clusters Visualization (t-SNE)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_file)
        logger.info(f"Visualization saved to {output_file}")
    
    def create_cluster_dataset(self, clusters, product=None):
        """
        Create a merged dataset with all contents matched with their cluster_id
        
        Args:
            clusters: Dictionary of clusters
            product: If specified, only create dataset for this product
            
        Returns:
            DataFrame with merged data
        """
        # Create a list of records
        records = []
        
        for cluster_id, items in clusters.items():
            for item in items:
                records.append({
                    "cluster_id": cluster_id,
                    "product": item["product"],
                    "category": item["category"],
                    "question_id": item["question_id"],
                    "question": item["question"],
                })
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        if product:
            # Filter by product if specified
            df = df[df["product"] == product]
        
        # Sort by cluster_id and product
        df = df.sort_values(["cluster_id", "product", "category"])
        
        # Save to CSV
        if product:
            output_file = os.path.join(self.clusters_dir, f"{product}_clusters.csv")
        else:
            output_file = os.path.join(self.clusters_dir, "all_clusters.csv")
        
        df.to_csv(output_file, index=False)
        logger.info(f"Cluster dataset saved to {output_file}")
        
        # Get additional data from original CSV files
        merged_df = self.merge_with_original_data(df)
        
        # Save merged dataset
        if product:
            merged_output_file = os.path.join(self.clusters_dir, f"{product}_merged_clusters.csv")
        else:
            merged_output_file = os.path.join(self.clusters_dir, "all_merged_clusters.csv")
        
        merged_df.to_csv(merged_output_file, index=False)
        logger.info(f"Merged cluster dataset saved to {merged_output_file}")
        
        return merged_df
    
    def merge_with_original_data(self, cluster_df):
        """
        Merge cluster data with original data from CSV files
        
        Args:
            cluster_df: DataFrame with cluster data
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging cluster data with original data")
        
        # Group by product and category
        grouped = cluster_df.groupby(["product", "category"])
        
        all_data = []
        
        for (product, category), group in tqdm(grouped, desc="Merging with original data"):
            # Load original data
            csv_path = os.path.join(self.output_dir, product, category, "data.csv")
            
            if not os.path.exists(csv_path):
                logger.warning(f"Original data file not found: {csv_path}")
                continue
            
            try:
                original_df = pd.read_csv(csv_path)
                
                # Merge with cluster data
                question_ids = group["question_id"].tolist()
                cluster_dict = {row["question_id"]: row["cluster_id"] for _, row in group.iterrows()}
                
                # Filter original data by question IDs
                filtered_df = original_df[original_df["id"].isin(question_ids)].copy()
                
                # Add cluster_id
                filtered_df["cluster_id"] = filtered_df["id"].map(cluster_dict)
                
                # Add product and category
                filtered_df["product"] = product
                filtered_df["category"] = category
                
                all_data.append(filtered_df)
            
            except Exception as e:
                logger.error(f"Error merging data for {product}/{category}: {str(e)}")
        
        if not all_data:
            logger.warning("No data merged")
            return cluster_df
        
        # Concatenate all data
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by cluster_id and product
        merged_df = merged_df.sort_values(["cluster_id", "product", "category"])
        
        logger.info(f"Merged data: {merged_df.shape[0]} rows")
        
        return merged_df
    
    def analyze_clusters(self, clusters):
        """
        Analyze clusters and print statistics
        
        Args:
            clusters: Dictionary of clusters
        """
        logger.info("Analyzing clusters")
        
        # Count items in each cluster
        cluster_sizes = {cluster_id: len(items) for cluster_id, items in clusters.items()}
        
        # Sort clusters by size
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        
        # Print statistics
        logger.info(f"Total clusters: {len(cluster_sizes)}")
        
        if -1 in cluster_sizes:
            noise_size = cluster_sizes[-1]
            logger.info(f"Noise points: {noise_size} ({noise_size/sum(cluster_sizes.values())*100:.2f}%)")
        
        # Top 10 largest clusters
        logger.info("Top 10 largest clusters:")
        for cluster_id, size in sorted_clusters[:10]:
            if cluster_id == -1:
                continue
            logger.info(f"Cluster {cluster_id}: {size} items")
        
        # Distribution of cluster sizes
        size_distribution = defaultdict(int)
        for _, size in sorted_clusters:
            if size >= 100:
                size_distribution["100+"] += 1
            elif size >= 50:
                size_distribution["50-99"] += 1
            elif size >= 20:
                size_distribution["20-49"] += 1
            elif size >= 10:
                size_distribution["10-19"] += 1
            elif size >= 5:
                size_distribution["5-9"] += 1
            else:
                size_distribution["<5"] += 1
        
        logger.info("Cluster size distribution:")
        for size_range, count in size_distribution.items():
            logger.info(f"{size_range}: {count} clusters")
    
    def run_pipeline(self, product=None):
        """
        Run the complete clustering pipeline
        
        Args:
            product: If specified, only process this product
            
        Returns:
            Merged DataFrame with cluster data
        """
        logger.info(f"Starting clustering pipeline")
        
        # Load embeddings
        embedding_data = self.load_embeddings(product)
        
        if not embedding_data:
            logger.error("No embeddings loaded. Exiting.")
            return None
        
        # Extract embeddings
        embeddings = [d["embedding"] for d in embedding_data]
        
        # Reduce dimensions
        reduced_embeddings = self.reduce_dimensions(embeddings, method='umap')
        
        # Cluster embeddings
        clusters, reduced_embeddings, cluster_labels = self.cluster_embeddings(reduced_embeddings, embedding_data)
        
        # Visualize clusters
        if self.visualize:
            vis_file = os.path.join(self.clusters_dir, "cluster_visualization.png")
            self.visualize_clusters(reduced_embeddings, cluster_labels, vis_file)
        
        # Analyze clusters
        self.analyze_clusters(clusters)
        
        # Create cluster dataset
        merged_df = self.create_cluster_dataset(clusters, product)
        
        logger.info("Clustering pipeline completed")
        
        return merged_df


def main():
    """
    Main function to run the clustering pipeline
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Clustering pipeline for embeddings")
    parser.add_argument("--models-dir", default="models", help="Directory containing embeddings and indices")
    parser.add_argument("--output-dir", default="output", help="Directory containing original data")
    parser.add_argument("--clusters-dir", default="clusters", help="Directory to save clustering results")
    parser.add_argument("--product", default=None, help="Process only specific product")
    parser.add_argument("--min-cluster-size", type=int, default=5, help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--min-samples", type=int, default=2, help="Minimum samples parameter for HDBSCAN")
    parser.add_argument("--umap-components", type=int, default=50, help="Number of components for UMAP dimensionality reduction")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="Number of neighbors for UMAP")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization generation")
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = ClusteringPipeline(
            models_dir=args.models_dir,
            output_dir=args.output_dir,
            clusters_dir=args.clusters_dir,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            umap_components=args.umap_components,
            umap_neighbors=args.umap_neighbors,
            visualize=not args.no_visualize
        )
        
        # Run pipeline
        pipeline.run_pipeline(args.product)
        
    except Exception as e:
        logger.error(f"Error running clustering pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
