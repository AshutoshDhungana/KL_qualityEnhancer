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
        logging.FileHandler("cache_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cache_generation")

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
    
    success = True
    for product in PRODUCTS:
        logger.info(f"======== Processing cache for {product} ========")
        
        # Run only the cluster cache generation step (step 5) for this product
        cmd = [
            sys.executable, 
            trigger_script, 
            "--product", product,
            "--start-step", "5",  # Start at step 5 (Cluster Cache Generation)
            "--end-step", "6"     # End at step 6 (exclusive)
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
    
    if success:
        logger.info("All cache generation completed successfully")
    else:
        logger.error("Some cache generation tasks failed")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 