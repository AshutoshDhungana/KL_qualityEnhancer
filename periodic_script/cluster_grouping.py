import os
import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import re
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("cluster_grouping.log"), logging.StreamHandler()]
)
logger = logging.getLogger("cluster_grouping")

def get_absolute_path(path):
    """Helper function to get absolute path from relative path"""
    if os.path.isabs(path):
        return path
    # Get the directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go one level up to the project root
    base_dir = os.path.dirname(current_dir)
    # Join with the relative path
    return os.path.join(base_dir, path)

class ClusterGroupingPipeline:
    def __init__(self, 
                 clusters_dir="clusters",
                 models_dir="models",
                 output_dir="processed_clusters",
                 canonical_selection="central_similarity"):
        """
        Initialize the cluster grouping pipeline
        
        Args:
            clusters_dir: Directory containing merged cluster results
            models_dir: Directory containing embeddings and indices
            output_dir: Directory to save processed results
            canonical_selection: Method to select canonical questions
                - "central_similarity": Select the most central/representative question (default)
                - "most_recent": Select the most recently created question
                - "earliest": Select the earliest created question
                - "longest": Select the question with the longest text
                - "shortest": Select the question with the shortest text
        """
        # Convert relative paths to absolute paths
        self.clusters_dir = get_absolute_path(clusters_dir)
        self.models_dir = get_absolute_path(models_dir)
        self.output_dir = get_absolute_path(output_dir)
        self.canonical_selection = canonical_selection
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load metadata if needed for embeddings
        self.metadata_path = os.path.join(self.models_dir, "embeddings_metadata.json")
        if os.path.exists(self.metadata_path) and self.canonical_selection == "central_similarity":
            try:
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded embeddings metadata from {self.metadata_path}")
            except Exception as e:
                logger.warning(f"Failed to load embeddings metadata: {str(e)}")
                self.metadata = None
        else:
            self.metadata = None
        
        logger.info(f"Initialized cluster grouping pipeline")
        logger.info(f"Clusters directory: {self.clusters_dir}")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Canonical selection method: {self.canonical_selection}")
    
    def load_merged_clusters(self, product=None):
        """
        Load merged clusters from CSV files
        
        Args:
            product: If specified, only load this product's clusters
            
        Returns:
            DataFrame with merged cluster data
        """
        try:
            if product:
                # Load specific product clusters
                file_path = os.path.join(self.clusters_dir, f"{product}_merged_clusters.csv")
                if not os.path.exists(file_path):
                    logger.error(f"Merged clusters file not found: {file_path}")
                    raise FileNotFoundError(f"Merged clusters file not found: {file_path}")
                
                logger.info(f"Loading merged clusters for product: {product} from {file_path}")
                
                try:
                    # Try to load the file
                    df = pd.read_csv(file_path)
                    logger.info(f"Loaded {len(df)} records")
                    return df
                except pd.errors.ParserError as e:
                    logger.error(f"Error parsing CSV file: {str(e)}")
                    raise
                except Exception as e:
                    logger.error(f"Error loading file: {str(e)}")
                    raise
            else:
                # Load all merged clusters
                file_path = os.path.join(self.clusters_dir, "all_merged_clusters.csv")
                if not os.path.exists(file_path):
                    # Try loading individual product files
                    logger.info("All merged clusters file not found, attempting to load individual product files")
                    
                    cluster_files = [f for f in os.listdir(self.clusters_dir) if f.endswith("_merged_clusters.csv")]
                    if not cluster_files:
                        logger.error(f"No merged cluster files found in: {self.clusters_dir}")
                        raise FileNotFoundError(f"No merged cluster files found in: {self.clusters_dir}")
                    
                    all_dfs = []
                    for file in cluster_files:
                        file_path = os.path.join(self.clusters_dir, file)
                        logger.info(f"Loading merged clusters from: {file}")
                        try:
                            df = pd.read_csv(file_path)
                            all_dfs.append(df)
                            logger.info(f"Loaded {len(df)} records from {file}")
                        except Exception as e:
                            logger.error(f"Error loading file {file}: {str(e)}")
                            # Continue with other files
                    
                    if not all_dfs:
                        logger.error("No files could be loaded")
                        raise ValueError("No files could be loaded")
                    
                    merged_df = pd.concat(all_dfs, ignore_index=True)
                    logger.info(f"Loaded {len(merged_df)} total records from {len(all_dfs)} files")
                    return merged_df
                else:
                    logger.info(f"Loading all merged clusters from {file_path}")
                    try:
                        df = pd.read_csv(file_path)
                        logger.info(f"Loaded {len(df)} records")
                        return df
                    except pd.errors.ParserError as e:
                        logger.error(f"Error parsing CSV file: {str(e)}")
                        raise
                    except Exception as e:
                        logger.error(f"Error loading file: {str(e)}")
                        raise
        except Exception as e:
            logger.error(f"Error in load_merged_clusters: {str(e)}")
            # Check if we can use a different approach
            if product:
                # Try loading the basic clusters file instead
                basic_file_path = os.path.join(self.clusters_dir, f"{product}_clusters.csv")
                if os.path.exists(basic_file_path):
                    logger.info(f"Attempting to load basic clusters file instead: {basic_file_path}")
                    try:
                        df = pd.read_csv(basic_file_path)
                        logger.info(f"Loaded {len(df)} records from basic clusters file")
                        logger.warning("Basic clusters file doesn't contain all metadata")
                        return df
                    except Exception as e2:
                        logger.error(f"Error loading basic clusters file: {str(e2)}")
            
            # Re-raise the original exception
            raise
    
    def load_embeddings(self, product, category, question_ids):
        """
        Load embeddings for a specific product/category/question_ids
        
        Args:
            product: Product name
            category: Category name
            question_ids: List of question IDs to load embeddings for
            
        Returns:
            Dictionary mapping question_id to embedding
        """
        if self.metadata is None:
            logger.debug("No metadata available, cannot load embeddings")
            return {}
        
        if product not in self.metadata["products"]:
            logger.debug(f"Product {product} not found in metadata")
            return {}
        
        if category not in self.metadata["products"][product]["categories"]:
            logger.debug(f"Category {category} not found in metadata for product {product}")
            return {}
        
        category_data = self.metadata["products"][product]["categories"][category]
        embeddings_file = category_data["embeddings_file"]
        embeddings_path = os.path.join(self.models_dir, product, embeddings_file)
        
        if not os.path.exists(embeddings_path):
            logger.debug(f"Embeddings file not found: {embeddings_path}")
            return {}
        
        try:
            # Load embeddings
            embeddings = np.load(embeddings_path)
            metadata_question_ids = category_data["question_ids"]
            
            # Create a mapping from question_id to embedding
            embedding_map = {}
            for q_id, emb in zip(metadata_question_ids, embeddings):
                if q_id in question_ids:
                    embedding_map[q_id] = emb
            
            logger.debug(f"Loaded {len(embedding_map)} embeddings for {product}/{category}")
            return embedding_map
        
        except Exception as e:
            logger.error(f"Error loading embeddings for {product}/{category}: {str(e)}")
            return {}
    
    def compute_similarity_scores(self, embeddings_list):
        """
        Compute pairwise cosine similarity between embeddings
        
        Args:
            embeddings_list: List of embeddings
            
        Returns:
            Pairwise similarity matrix
        """
        if not embeddings_list:
            return np.array([])
        
        # Convert list to numpy array
        embeddings_array = np.array(embeddings_list)
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(embeddings_array)
        
        return similarity_matrix
    
    def identify_central_question(self, group, embeddings=None):
        """
        Identify the most central/representative question in a cluster
        
        Args:
            group: DataFrame with questions in a cluster
            embeddings: Dictionary mapping question_id to embedding (optional)
            
        Returns:
            Index of the central question
        """
        if len(group) == 1:
            # If there's only one question, it's the central one
            return group.index[0]
        
        if embeddings is None or len(embeddings) < 2:
            # Fall back to most recent if embeddings not available
            if pd.api.types.is_string_dtype(group['created_at']):
                group['created_at'] = pd.to_datetime(group['created_at'])
            return group['created_at'].idxmax()
        
        # Get the embeddings in the same order as the group rows
        embeddings_list = []
        valid_indices = []
        
        for idx, row in group.iterrows():
            q_id = row['id'] if 'id' in row else row['question_id']
            if q_id in embeddings:
                embeddings_list.append(embeddings[q_id])
                valid_indices.append(idx)
        
        if len(embeddings_list) < 2:
            # Not enough embeddings to compute centrality
            if pd.api.types.is_string_dtype(group['created_at']):
                group['created_at'] = pd.to_datetime(group['created_at'])
            return group['created_at'].idxmax()
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_scores(embeddings_list)
        
        # Compute average similarity to all other questions
        avg_similarities = np.mean(similarity_matrix, axis=1)
        
        # Get the index of the most central question
        central_idx = np.argmax(avg_similarities)
        
        # Map back to the original DataFrame index
        return valid_indices[central_idx]
    
    def identify_canonical_questions(self, df):
        """
        Identify canonical questions in each cluster
        
        Args:
            df: DataFrame with merged cluster data
            
        Returns:
            DataFrame with canonical and redundant flags
        """
        logger.info("Identifying canonical questions in clusters")
        
        # Make a copy of the DataFrame to avoid modifying the original
        processed_df = df.copy()
        
        # Add is_canonical and is_redundant columns, initialize to False
        processed_df['is_canonical'] = False
        processed_df['is_redundant'] = False
        
        # Skip noise points (cluster_id == -1)
        valid_clusters = processed_df[processed_df['cluster_id'] != -1]
        
        # If no valid clusters, return the original DataFrame with default flags
        if len(valid_clusters) == 0:
            logger.warning("No valid clusters found (all points are noise)")
            return processed_df
        
        # Check if we have the required columns for central similarity
        has_product_category = 'product' in valid_clusters.columns and 'category' in valid_clusters.columns
        
        # If using central similarity and we have the required metadata
        if self.canonical_selection == "central_similarity" and self.metadata is not None and has_product_category:
            logger.info("Using central similarity method for canonical selection")
            
            try:
                # First, group by product and category to load embeddings efficiently
                product_category_groups = valid_clusters.groupby(['product', 'category'])
                
                for (product, category), pc_group in tqdm(product_category_groups, desc="Processing product/category groups"):
                    # Get question IDs for this product/category
                    id_column = 'id' if 'id' in pc_group.columns else 'question_id'
                    if id_column not in pc_group.columns:
                        logger.warning(f"No question ID column found for {product}/{category}, skipping")
                        continue
                    
                    question_ids = pc_group[id_column].tolist()
                    
                    # Load embeddings for these questions
                    embeddings = self.load_embeddings(product, category, question_ids)
                    
                    # Process each cluster within this product/category
                    for cluster_id, cluster_group in pc_group.groupby('cluster_id'):
                        canonical_idx = self.identify_central_question(cluster_group, embeddings)
                        
                        # Mark canonical question
                        processed_df.at[canonical_idx, 'is_canonical'] = True
                        
                        # Mark all other questions in the cluster as redundant
                        redundant_indices = cluster_group.index.difference([canonical_idx])
                        processed_df.loc[redundant_indices, 'is_redundant'] = True
            
            except Exception as e:
                logger.error(f"Error in central similarity processing: {str(e)}")
                logger.info("Falling back to simple method for canonical selection")
                
                # Process without embeddings
                grouped = valid_clusters.groupby('cluster_id')
                
                for cluster_id, group in tqdm(grouped, desc="Processing clusters"):
                    canonical_idx = self.select_canonical_by_method(group)
                    
                    # Mark canonical question
                    processed_df.at[canonical_idx, 'is_canonical'] = True
                    
                    # Mark all other questions in the cluster as redundant
                    redundant_indices = group.index.difference([canonical_idx])
                    processed_df.loc[redundant_indices, 'is_redundant'] = True
        else:
            # Process without embeddings
            if self.canonical_selection == "central_similarity":
                if not has_product_category:
                    logger.warning("Missing product or category columns, using alternative method")
                elif self.metadata is None:
                    logger.warning("No embeddings metadata available, using alternative method")
                
                logger.info(f"Using {self.canonical_selection} method for canonical selection")
            
            grouped = valid_clusters.groupby('cluster_id')
            
            for cluster_id, group in tqdm(grouped, desc="Processing clusters"):
                canonical_idx = self.select_canonical_by_method(group)
                
                # Mark canonical question
                processed_df.at[canonical_idx, 'is_canonical'] = True
                
                # Mark all other questions in the cluster as redundant
                redundant_indices = group.index.difference([canonical_idx])
                processed_df.loc[redundant_indices, 'is_redundant'] = True
        
        # Count canonical and redundant questions
        n_canonical = processed_df['is_canonical'].sum()
        n_redundant = processed_df['is_redundant'].sum()
        logger.info(f"Identified {n_canonical} canonical questions and {n_redundant} redundant questions")
        
        return processed_df
    
    def select_canonical_by_method(self, group):
        """
        Select canonical question based on selection method
        
        Args:
            group: DataFrame with questions in a cluster
            
        Returns:
            Index of the selected canonical question
        """
        if len(group) == 1:
            # If there's only one question in the cluster, it's canonical
            return group.index[0]
        
        if self.canonical_selection == "most_recent":
            # Convert created_at to datetime if it's not already
            if pd.api.types.is_string_dtype(group['created_at']):
                group['created_at'] = pd.to_datetime(group['created_at'])
            # Return the index of the most recent question
            return group['created_at'].idxmax()
        
        elif self.canonical_selection == "earliest":
            # Convert created_at to datetime if it's not already
            if pd.api.types.is_string_dtype(group['created_at']):
                group['created_at'] = pd.to_datetime(group['created_at'])
            # Return the index of the earliest question
            return group['created_at'].idxmin()
        
        elif self.canonical_selection == "longest":
            # Return the index of the question with the longest text
            return group['question'].str.len().idxmax()
        
        elif self.canonical_selection == "shortest":
            # Return the index of the question with the shortest text
            return group['question'].str.len().idxmin()
        
        else:
            # Default to most recent
            if pd.api.types.is_string_dtype(group['created_at']):
                group['created_at'] = pd.to_datetime(group['created_at'])
            return group['created_at'].idxmax()
    
    def mark_archived_questions(self, df):
        """
        Mark questions as archived or active based on deleted_at field
        
        Args:
            df: DataFrame with merged cluster data
            
        Returns:
            DataFrame with archived and active flags
        """
        logger.info("Marking archived and active questions")
        
        # Make a copy of the DataFrame to avoid modifying the original
        processed_df = df.copy()
        
        # Add is_archived and is_active columns
        processed_df['is_archived'] = False
        processed_df['is_active'] = False
        
        # Check if deleted_at column exists
        if 'deleted_at' not in processed_df.columns:
            logger.warning("deleted_at column not found, assuming all questions are active")
            processed_df['is_active'] = True
            return processed_df
        
        # Mark questions as archived if deleted_at is not empty/null
        # First, convert NaN to empty string
        processed_df['deleted_at'] = processed_df['deleted_at'].fillna('')
        
        # Mark as archived if deleted_at is not empty
        processed_df['is_archived'] = processed_df['deleted_at'] != ''
        
        # Mark as active if not archived
        processed_df['is_active'] = ~processed_df['is_archived']
        
        # Count archived and active questions
        n_archived = processed_df['is_archived'].sum()
        n_active = processed_df['is_active'].sum()
        logger.info(f"Identified {n_archived} archived questions and {n_active} active questions")
        
        return processed_df
    
    def save_processed_data(self, df, product=None):
        """
        Save processed data to CSV files
        
        Args:
            df: DataFrame with processed data
            product: If specified, save with product name prefix
            
        Returns:
            Path to saved CSV file
        """
        if product:
            output_file = os.path.join(self.output_dir, f"{product}_processed_clusters.csv")
        else:
            output_file = os.path.join(self.output_dir, "all_processed_clusters.csv")
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Processed data saved to {output_file}")
        
        # Generate summary statistics
        self.generate_summary(df, product)
        
        return output_file
    
    def generate_summary(self, df, product=None):
        """
        Generate summary statistics for processed data
        
        Args:
            df: DataFrame with processed data
            product: Product name for the summary
        """
        summary = {
            "product": product if product else "all",
            "total_questions": len(df),
            "clusters": {
                "total": len(df[df['cluster_id'] != -1]['cluster_id'].unique()),
                "noise_points": len(df[df['cluster_id'] == -1])
            },
            "questions": {
                "canonical": int(df['is_canonical'].sum()),
                "redundant": int(df['is_redundant'].sum()),
                "archived": int(df['is_archived'].sum() if 'is_archived' in df.columns else 0),
                "active": int(df['is_active'].sum() if 'is_active' in df.columns else 0),
                "active_canonical": int((df['is_canonical'] & df['is_active']).sum() if 'is_active' in df.columns else 0),
                "archived_canonical": int((df['is_canonical'] & df['is_archived']).sum() if 'is_archived' in df.columns else 0)
            }
        }
        
        # Add cluster size distribution
        if len(df[df['cluster_id'] != -1]) > 0:
            cluster_sizes = df[df['cluster_id'] != -1].groupby('cluster_id').size()
            summary["cluster_size_distribution"] = {
                "min": int(cluster_sizes.min()),
                "max": int(cluster_sizes.max()),
                "mean": float(cluster_sizes.mean()),
                "median": float(cluster_sizes.median())
            }
        
        # Save summary to JSON
        if product:
            summary_file = os.path.join(self.output_dir, f"{product}_summary.json")
        else:
            summary_file = os.path.join(self.output_dir, "all_summary.json")
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary statistics saved to {summary_file}")
    
    def run_pipeline(self, product=None):
        """
        Run the complete cluster grouping pipeline
        
        Args:
            product: If specified, only process this product
            
        Returns:
            DataFrame with processed data
        """
        logger.info(f"Starting cluster grouping pipeline")
        
        # Load merged clusters
        df = self.load_merged_clusters(product)
        
        # Identify canonical questions
        df = self.identify_canonical_questions(df)
        
        # Mark archived questions
        df = self.mark_archived_questions(df)
        
        # Save processed data
        self.save_processed_data(df, product)
        
        logger.info("Cluster grouping pipeline completed")
        
        return df


def main():
    """
    Main function to run the cluster grouping pipeline
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cluster grouping pipeline for processed clusters")
    parser.add_argument("--clusters-dir", default="clusters", help="Directory containing merged cluster results")
    parser.add_argument("--models-dir", default="models", help="Directory containing embeddings and indices")
    parser.add_argument("--output-dir", default="processed_clusters", help="Directory to save processed results")
    parser.add_argument("--product", default=None, help="Process only specific product")
    parser.add_argument("--canonical-selection", default="central_similarity", 
                        choices=["central_similarity", "most_recent", "earliest", "longest", "shortest"],
                        help="Method to select canonical questions")
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = ClusterGroupingPipeline(
            clusters_dir=args.clusters_dir,
            models_dir=args.models_dir,
            output_dir=args.output_dir,
            canonical_selection=args.canonical_selection
        )
        
        # Run pipeline
        pipeline.run_pipeline(args.product)
        
    except Exception as e:
        logger.error(f"Error running cluster grouping pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
