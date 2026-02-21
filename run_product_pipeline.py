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
        logging.FileHandler("product_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("product_pipeline")

# List of products to process
PRODUCTS = [
    "Danfe_Corp_Product_1",
    "Danfe_Corp_Product_2",
    "Danfe_Corp_Product_3",
    "Danfe_Corp_Product_4"
]

def get_project_root():
    """Get the project root directory"""
    return Path(os.path.dirname(os.path.abspath(__file__)))

def run_trigger_for_product(product, start_step=0):
    """Run the trigger script for a specific product"""
    try:
        logger.info(f"Running pipeline for {product}")
        
        project_root = get_project_root()
        trigger_script = project_root / "periodic_script" / "trigger.py"
        
        if not trigger_script.exists():
            logger.error(f"Trigger script not found at: {trigger_script}")
            return False
        
        # Run the trigger script
        cmd = [
            sys.executable,
            str(trigger_script),
            "--product", product,
            "--start-step", str(start_step)
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Output for {product}:\n{result.stdout}")
        logger.info(f"Successfully completed pipeline for {product}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing {product}: {e}")
        logger.error(f"Command output: {e.stdout}")
        logger.error(f"Command error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing {product}: {e}")
        return False

def run_cache_generation():
    """Run the combined cache generation script"""
    try:
        logger.info("Running combined cache generation")
        
        project_root = get_project_root()
        cache_script = project_root / "run_combined_cache_generation.py"
        
        if not cache_script.exists():
            logger.error(f"Cache generation script not found at: {cache_script}")
            return False
        
        # Run the cache generation script
        cmd = [sys.executable, str(cache_script)]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Output from cache generation:\n{result.stdout}")
        logger.info("Successfully completed cache generation")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running cache generation: {e}")
        logger.error(f"Command output: {e.stdout}")
        logger.error(f"Command error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running cache generation: {e}")
        return False

def generate_product_summaries():
    """Generate product-specific summary files"""
    try:
        logger.info("Generating product summaries")
        
        project_root = get_project_root()
        summary_script = project_root / "generate_product_summaries.py"
        
        if not summary_script.exists():
            logger.error(f"Summary generation script not found at: {summary_script}")
            return False
        
        # Run the summary generation script
        cmd = [sys.executable, str(summary_script)]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Output from summary generation:\n{result.stdout}")
        logger.info("Successfully completed product summary generation")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating product summaries: {e}")
        logger.error(f"Command output: {e.stdout}")
        logger.error(f"Command error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error generating product summaries: {e}")
        return False

def main():
    logger.info("Starting product pipeline")
    
    # Track overall success
    success = True
    
    # Step 1: Run pipeline for each product
    for product in PRODUCTS:
        logger.info(f"======== Processing {product} ========")
        product_success = run_trigger_for_product(product)
        if not product_success:
            logger.error(f"Failed to process {product}")
            success = False
        logger.info(f"======== Completed {product} ========\n")
    
    # Step 2: Generate product-specific cluster cache files
    logger.info("======== Running Cache Generation ========")
    cache_success = run_cache_generation()
    if not cache_success:
        logger.error("Failed to generate cache files")
        success = False
    logger.info("======== Completed Cache Generation ========\n")
    
    # Step 3: Generate product-specific summary files
    logger.info("======== Generating Product Summaries ========")
    summary_success = generate_product_summaries()
    if not summary_success:
        logger.error("Failed to generate product summaries")
        success = False
    logger.info("======== Completed Product Summaries ========\n")
    
    if success:
        logger.info("All product pipeline steps completed successfully")
    else:
        logger.error("Some product pipeline steps failed")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 