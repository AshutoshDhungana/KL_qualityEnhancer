#!/usr/bin/env python3
import os
import sys
import subprocess
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trigger_products.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trigger_products")

# List of products to process
PRODUCTS = [
    "Danfe Corp Product 1",
    "Danfe Corp Product 2",
    "Danfe Corp Product 3",
    "Danfe Corp Product 4"
]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trigger_script = os.path.join(script_dir, "periodic_script", "trigger.py")
    
    # Verify trigger script exists
    if not os.path.exists(trigger_script):
        logger.error(f"Trigger script not found at: {trigger_script}")
        return 1
    
    overall_success = True
    
    for product in PRODUCTS:
        logger.info(f"=== Running full pipeline for {product} ===")
        start_time = time.time()
        
        # Run the trigger script for this product
        cmd = [
            sys.executable, 
            trigger_script, 
            "--product", product
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            # Run the command and capture output
            result = subprocess.run(
                cmd, 
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Log the output
            if result.stdout:
                logger.info(f"Output for {product}:\n{result.stdout}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Successfully completed pipeline for {product} in {elapsed_time:.2f} seconds")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error processing {product}: {e}")
            logger.error(f"Command output: {e.stdout}")
            logger.error(f"Command error: {e.stderr}")
            overall_success = False
        except Exception as e:
            logger.error(f"Unexpected error processing {product}: {e}")
            overall_success = False
        
        logger.info(f"=== Completed {product} ===\n")
    
    # Run for all products combined
    logger.info("=== Running full pipeline for all products ===")
    start_time = time.time()
    
    cmd = [
        sys.executable, 
        trigger_script
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Log the output
        if result.stdout:
            logger.info(f"Output for all products:\n{result.stdout}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully completed pipeline for all products in {elapsed_time:.2f} seconds")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing all products: {e}")
        logger.error(f"Command output: {e.stdout}")
        logger.error(f"Command error: {e.stderr}")
        overall_success = False
    except Exception as e:
        logger.error(f"Unexpected error processing all products: {e}")
        overall_success = False
    
    logger.info("=== Completed all products ===\n")
    
    if overall_success:
        logger.info("All pipeline runs completed successfully")
    else:
        logger.error("Some pipeline runs failed")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main()) 