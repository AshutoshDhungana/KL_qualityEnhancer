#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import subprocess
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_trigger.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pipeline_trigger")

class PipelineTrigger:
    def __init__(self, product=None):
        self.product = product
        self.start_time = time.time()
        self.src_dir = os.path.dirname(os.path.abspath(__file__))
        self.steps = [
            {
                "name": "Timed.py",
                "file": "timed.py",
                "args": []
            },
            {
                "name": "Embedding",
                "file": "embedding.py",
                "args": []
            },
            {
                "name": "Clustering",
                "file": "clustering.py",
                "args": []
            },
            {
                "name": "Cluster Grouping",
                "file": "cluster_grouping.py",
                "args": ["--product", product] if product else []
            },
            {
                "name": "Cleaned Dataset Creation",
                "file": "cleaned_ds_create.py",
                "args": ["--product", product] if product else []
            },
            {
                "name": "Cluster Cache Generation",
                "file": "cluster_cache.py",
                "args": ["--input", f"{product}_complete_dataset.csv" if product else "all_complete_dataset.csv"]
            }
        ]
    
    def run_step(self, step, dry_run=False):
        """Run a single step of the pipeline"""
        file_path = os.path.join(self.src_dir, step["file"])
        cmd = [sys.executable, file_path] + step["args"]
        
        logger.info(f"Running {step['name']}: {' '.join(cmd)}")
        
        if dry_run:
            logger.info(f"DRY RUN: Would execute {' '.join(cmd)}")
            return True
        
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
                logger.info(f"Output from {step['name']}:\n{result.stdout}")
            
            logger.info(f"Successfully completed {step['name']}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running {step['name']}: {e}")
            logger.error(f"Command output: {e.stdout}")
            logger.error(f"Command error: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error running {step['name']}: {e}")
            return False
    
    def run_pipeline(self, start_step=0, end_step=None, dry_run=False):
        """Run the complete pipeline or a subset of steps"""
        if end_step is None:
            end_step = len(self.steps)
        
        logger.info(f"Starting pipeline at step {start_step} and ending at step {end_step}")
        if self.product:
            logger.info(f"Running for product: {self.product}")
        
        success = True
        for i, step in enumerate(self.steps[start_step:end_step], start=start_step):
            logger.info(f"Step {i+1}/{end_step}: {step['name']}")
            step_success = self.run_step(step, dry_run)
            
            if not step_success:
                logger.error(f"Pipeline failed at step {i+1}: {step['name']}")
                success = False
                break
        
        elapsed_time = time.time() - self.start_time
        if success:
            logger.info(f"Pipeline completed successfully in {elapsed_time:.2f} seconds")
        else:
            logger.error(f"Pipeline failed after {elapsed_time:.2f} seconds")
        
        return success

def main():
    parser = argparse.ArgumentParser(description="Run the complete SecurityPal pipeline")
    parser.add_argument("--product", help="Process only a specific product")
    parser.add_argument("--start-step", type=int, default=0, help="Start at this step (0-based index)")
    parser.add_argument("--end-step", type=int, help="End at this step (0-based index)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be executed without running anything")
    parser.add_argument("--list-steps", action="store_true", help="List available pipeline steps and exit")
    args = parser.parse_args()
    
    pipeline = PipelineTrigger(args.product)
    
    if args.list_steps:
        print("Available pipeline steps:")
        for i, step in enumerate(pipeline.steps):
            print(f"{i}: {step['name']} ({step['file']})")
        return 0
    
    success = pipeline.run_pipeline(
        start_step=args.start_step,
        end_step=args.end_step,
        dry_run=args.dry_run
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
