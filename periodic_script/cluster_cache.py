#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import logging
import json
import os
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the project root directory"""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent

def ensure_directories():
    """Create necessary directories if they don't exist"""
    project_root = get_project_root()
    directories = {
        'cache': project_root / "cache",
        'processed': project_root / "processed_clusters",
        'cleaned': project_root / "cleaned_dataset"
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")
    
    return directories

def initialize_summary_file(processed_dir, df, cluster_embeddings):
    """Initialize the summary file with statistics from the data"""
    summary_file = processed_dir / "all_summary.json"
    
    # Calculate cluster statistics
    cluster_sizes = df.groupby('cluster_id').size()
    total_clusters = len(cluster_embeddings)
    noise_points = len(df[df['cluster_id'] == -1]) if 'cluster_id' in df.columns else 0
    
    # Calculate question statistics
    canonical_mask = df['is_canonical'] if 'is_canonical' in df.columns else pd.Series(False, index=df.index)
    archived_mask = df['is_archived'] if 'is_archived' in df.columns else pd.Series(False, index=df.index)
    
    summary = {
        "product": "all",
        "total_questions": len(df),
        "clusters": {
            "total": total_clusters,
            "noise_points": noise_points
        },
        "questions": {
            "canonical": sum(canonical_mask),
            "redundant": len(df) - sum(canonical_mask),
            "archived": sum(archived_mask),
            "active": len(df) - sum(archived_mask),
            "active_canonical": sum(canonical_mask & ~archived_mask),
            "archived_canonical": sum(canonical_mask & archived_mask)
        },
        "cluster_size_distribution": {
            "min": int(cluster_sizes.min()) if not cluster_sizes.empty else 0,
            "max": int(cluster_sizes.max()) if not cluster_sizes.empty else 0,
            "mean": float(cluster_sizes.mean()) if not cluster_sizes.empty else 0,
            "median": float(cluster_sizes.median()) if not cluster_sizes.empty else 0
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Initialized summary file: {summary_file}")
    return summary_file

def initialize_clusters_file(processed_dir, cluster_embeddings, df):
    """Initialize the clusters file with data from embeddings"""
    clusters_file = processed_dir / "all_clusters.json"
    
    clusters_data = {
        "clusters": []
    }
    
    for _, row in cluster_embeddings.iterrows():
        cluster_id = int(row['cluster_id'])
        cluster_df = df[df['cluster_id'] == cluster_id]
        
        cluster = {
            "cluster_id": cluster_id,
            "embedding": row['embedding'].tolist(),
            "size": len(cluster_df),
            "questions": cluster_df['question'].tolist() if 'question' in cluster_df.columns else [],
            "canonical_questions": cluster_df[cluster_df['is_canonical']]['question'].tolist() 
                if 'is_canonical' in cluster_df.columns else [],
            "health_status": "Needs Review",
            "last_modified": datetime.now().isoformat(),
            "last_reviewed": None
        }
        clusters_data["clusters"].append(cluster)
    
    with open(clusters_file, 'w') as f:
        json.dump(clusters_data, f, indent=2)
    logger.info(f"Initialized clusters file: {clusters_file}")
    return clusters_file

def load_cluster_data(input_file):
    """Load the clustering results from CSV file"""
    try:
        project_root = get_project_root()
        input_path = project_root / "cleaned_dataset" / input_file
        
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found at: {input_path}\n"
                f"Please ensure the file exists in the cleaned_dataset directory."
            )
            
        logger.info(f"Loading cluster data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Verify required columns exist
        required_columns = ['cluster_id', 'embedding']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Input file is missing required columns: {', '.join(missing_columns)}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading cluster data: {e}")
        raise

def extract_cluster_embeddings(df):
    """Extract average embeddings for each cluster"""
    logger.info("Extracting cluster embeddings")
    
    try:
        # Convert string representation of embedding back to list
        df['embedding'] = df['embedding'].apply(eval)
        
        # Group by cluster and calculate mean embedding
        cluster_embeddings = df.groupby('cluster_id')['embedding'].apply(
            lambda x: np.mean(np.vstack(x), axis=0)
        ).reset_index()
        
        return cluster_embeddings
    except Exception as e:
        logger.error(f"Error extracting cluster embeddings: {e}")
        raise

def save_cache(cluster_embeddings, output_file):
    """Save cluster embeddings to cache file"""
    logger.info(f"Saving cluster cache to {output_file}")
    
    try:
        cache_data = {
            'clusters': []
        }
        
        for _, row in cluster_embeddings.iterrows():
            cache_data['clusters'].append({
                'cluster_id': int(row['cluster_id']),
                'embedding': row['embedding'].tolist()
            })
        
        # Ensure the output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving cache: {e}")
        raise

def save_individual_embeddings(df, cache_dir):
    """Save individual question embeddings for search functionality"""
    logger.info("Saving individual embeddings cache")
    
    try:
        # Create embeddings cache
        embeddings_cache = {
            'embeddings': [],
            'metadata': {
                'total_embeddings': len(df),
                'embedding_size': len(eval(df.iloc[0]['embedding'])) if len(df) > 0 else 0,
                'last_updated': datetime.now().isoformat()
            }
        }
        
        # Process each row
        for idx, row in df.iterrows():
            embedding_data = {
                'id': idx,
                'question': row['question'] if 'question' in df.columns else '',
                'cluster_id': int(row['cluster_id']),
                'embedding': eval(row['embedding']),
                'is_canonical': bool(row['is_canonical']) if 'is_canonical' in df.columns else False,
                'is_archived': bool(row['is_archived']) if 'is_archived' in df.columns else False
            }
            embeddings_cache['embeddings'].append(embedding_data)
        
        # Save to file
        cache_file = cache_dir / "question_embeddings.json"
        with open(cache_file, 'w') as f:
            json.dump(embeddings_cache, f, indent=2)
        
        # Save a numpy version for faster loading
        embeddings_array = np.array([e['embedding'] for e in embeddings_cache['embeddings']])
        np_cache_file = cache_dir / "question_embeddings.npy"
        np.save(np_cache_file, embeddings_array)
        
        # Save metadata separately for quick access
        metadata_file = cache_dir / "embeddings_metadata.json"
        metadata = {
            'question_ids': [e['id'] for e in embeddings_cache['embeddings']],
            'questions': [e['question'] for e in embeddings_cache['embeddings']],
            'cluster_ids': [e['cluster_id'] for e in embeddings_cache['embeddings']],
            'is_canonical': [e['is_canonical'] for e in embeddings_cache['embeddings']],
            'is_archived': [e['is_archived'] for e in embeddings_cache['embeddings']],
            **embeddings_cache['metadata']
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully saved individual embeddings cache with {len(df)} entries")
        return True
    except Exception as e:
        logger.error(f"Error saving individual embeddings: {e}")
        return False

def load_individual_embeddings(cache_dir):
    """Load individual embeddings for similarity search"""
    try:
        # Load numpy array of embeddings
        np_cache_file = cache_dir / "question_embeddings.npy"
        embeddings_array = np.load(np_cache_file)
        
        # Load metadata
        metadata_file = cache_dir / "embeddings_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return embeddings_array, metadata
    except Exception as e:
        logger.error(f"Error loading individual embeddings: {e}")
        return None, None

def find_similar_questions(query_embedding, cache_dir, top_k=5):
    """Find similar questions using cosine similarity"""
    try:
        # Load embeddings and metadata
        embeddings_array, metadata = load_individual_embeddings(cache_dir)
        if embeddings_array is None or metadata is None:
            return None
        
        # Calculate cosine similarity
        query_embedding = np.array(query_embedding)
        similarities = np.dot(embeddings_array, query_embedding) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k similar questions
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'question': metadata['questions'][idx],
                'similarity': float(similarities[idx]),
                'cluster_id': metadata['cluster_ids'][idx],
                'is_canonical': metadata['is_canonical'][idx],
                'is_archived': metadata['is_archived'][idx]
            })
        
        return results
    except Exception as e:
        logger.error(f"Error finding similar questions: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate cluster embedding cache")
    parser.add_argument(
        "--input",
        default="all_complete_dataset.csv",
        help="Input CSV file name in cleaned_dataset directory (default: all_complete_dataset.csv)"
    )
    parser.add_argument(
        "--output",
        help="Output JSON file for cluster cache (default: cache/cluster_cache.json)"
    )
    
    args = parser.parse_args()
    
    try:
        # Ensure all necessary directories exist
        directories = ensure_directories()
        
        # Set default output path if not provided
        if not args.output:
            args.output = directories['cache'] / "cluster_cache.json"
        else:
            # If output path is provided, make it relative to project root
            args.output = get_project_root() / args.output
        
        # Load cluster data
        df = load_cluster_data(args.input)
        
        # Extract cluster embeddings
        cluster_embeddings = extract_cluster_embeddings(df)
        
        # Initialize or update necessary files
        initialize_summary_file(directories['processed'], df, cluster_embeddings)
        initialize_clusters_file(directories['processed'], cluster_embeddings, df)
        
        # Save cluster cache
        save_cache(cluster_embeddings, args.output)
        
        # Save individual embeddings cache
        save_individual_embeddings(df, directories['cache'])
        
        logger.info("Cache generation completed successfully")
        
    except Exception as e:
        logger.error(f"Cache generation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
