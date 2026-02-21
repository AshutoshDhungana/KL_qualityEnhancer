#!/usr/bin/env python3
import os
import sys
import subprocess
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("combined_cache_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("combined_cache_generation")

# List of products to process (using correct names with underscores)
PRODUCTS = [
    "Danfe_Corp_Product_1",
    "Danfe_Corp_Product_2",
    "Danfe_Corp_Product_3",
    "Danfe_Corp_Product_4"
]

def get_project_root():
    """Get the project root directory"""
    return Path(os.path.dirname(os.path.abspath(__file__)))

def filter_dataset_by_product(df, product_name):
    """Filter the combined dataset to only include entries for a specific product"""
    if 'product' not in df.columns:
        logger.error(f"Dataset does not contain a 'product' column. Available columns: {df.columns.tolist()}")
        return None
    
    # Filter dataset for the specific product
    product_df = df[df['product'] == product_name].copy()
    
    if len(product_df) == 0:
        logger.warning(f"No entries found for product '{product_name}' in the dataset")
        return None
    
    logger.info(f"Filtered dataset for {product_name}: {len(product_df)} entries")
    return product_df

def run_cluster_cache_for_product(product_name, temp_file_path, cache_script):
    """Run the cluster cache script for a specific product using a temporary CSV file"""
    try:
        logger.info(f"Running cluster cache generation for {product_name}")
        
        # Run the cluster cache generation script
        cmd = [
            sys.executable, 
            str(cache_script),
            "--input", os.path.basename(str(temp_file_path))
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Output for {product_name}:\n{result.stdout}")
        logger.info(f"Successfully completed cache generation for {product_name}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing {product_name}: {e}")
        logger.error(f"Command output: {e.stdout}")
        logger.error(f"Command error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing {product_name}: {e}")
        return False

def main():
    # Initialize paths
    project_root = get_project_root()
    cache_script = project_root / "periodic_script" / "cluster_cache.py"
    combined_dataset_path = project_root / "cleaned_dataset" / "all_complete_dataset.csv"
    
    # Verify paths exist
    if not cache_script.exists():
        logger.error(f"Cache script not found at: {cache_script}")
        return 1
    
    if not combined_dataset_path.exists():
        logger.error(f"Combined dataset not found at: {combined_dataset_path}")
        return 1
    
    # Load the combined dataset
    try:
        logger.info(f"Loading combined dataset from {combined_dataset_path}")
        combined_df = pd.read_csv(combined_dataset_path)
        logger.info(f"Loaded dataset with {len(combined_df)} entries")
    except Exception as e:
        logger.error(f"Error loading combined dataset: {e}")
        return 1
    
    # Process each product
    success = True
    for product in PRODUCTS:
        logger.info(f"======== Processing {product} ========")
        
        # Filter the dataset for this product
        product_df = filter_dataset_by_product(combined_df, product)
        if product_df is None:
            logger.error(f"Skipping cache generation for {product} due to filtering error")
            success = False
            continue
        
        # Create a temporary file with the filtered dataset
        temp_file_name = f"{product}_complete_dataset.csv"
        temp_file_path = project_root / "cleaned_dataset" / temp_file_name
        
        try:
            # Save the filtered dataset
            product_df.to_csv(temp_file_path, index=False)
            logger.info(f"Saved filtered dataset to {temp_file_path}")
            
            # Run cluster cache generation
            product_success = run_cluster_cache_for_product(product, temp_file_path, cache_script)
            if not product_success:
                success = False
            
        except Exception as e:
            logger.error(f"Error processing {product}: {e}")
            success = False
        finally:
            # Clean up temporary file
            if temp_file_path.exists():
                logger.info(f"Keeping temporary dataset file for future use: {temp_file_path}")
            
        logger.info(f"======== Completed {product} ========\n")
    
    # Finally, process the combined dataset
    logger.info("======== Processing all products combined ========")
    
    try:
        # Run cluster cache generation for all products combined
        cmd = [
            sys.executable, 
            str(cache_script),
            "--input", "all_complete_dataset.csv"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Output for combined dataset:\n{result.stdout}")
        logger.info("Successfully completed cache generation for combined dataset")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing combined dataset: {e}")
        logger.error(f"Command output: {e.stdout}")
        logger.error(f"Command error: {e.stderr}")
        success = False
    except Exception as e:
        logger.error(f"Unexpected error processing combined dataset: {e}")
        success = False
    
    logger.info("======== Completed combined dataset ========\n")
    
    if success:
        logger.info("All cache generation completed successfully")
    else:
        logger.error("Some cache generation tasks failed")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 