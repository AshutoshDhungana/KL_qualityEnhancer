#!/usr/bin/env python3
import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("direct_cache_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("direct_cache_generation")

# List of products to process
PRODUCTS = [
    "Danfe Corp Product 1",
    "Danfe Corp Product 2",
    "Danfe Corp Product 3",
    "Danfe Corp Product 4"
]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_script = os.path.join(script_dir, "periodic_script", "cluster_cache.py")
    
    # Verify cache script exists
    if not os.path.exists(cache_script):
        logger.error(f"Cache script not found at: {cache_script}")
        return 1
    
    # Process all products plus the combined dataset
    success = True
    
    # First, process all products individually
    for product in PRODUCTS:
        product_file = f"{product.replace(' ', '_')}_complete_dataset.csv"
        logger.info(f"======== Processing cache for {product} ========")
        
        # Run the cluster cache generation script directly
        cmd = [
            sys.executable, 
            cache_script, 
            "--input", product_file
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"Output for {product}:\n{result.stdout}")
            logger.info(f"Successfully completed cache generation for {product}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error processing {product}: {e}")
            logger.error(f"Command output: {e.stdout}")
            logger.error(f"Command error: {e.stderr}")
            success = False
        except Exception as e:
            logger.error(f"Unexpected error processing {product}: {e}")
            success = False
            
        logger.info(f"======== Completed {product} ========\n")
    
    # Finally, process the combined dataset
    logger.info("======== Processing cache for all products combined ========")
    
    cmd = [
        sys.executable, 
        cache_script, 
        "--input", "all_complete_dataset.csv"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
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