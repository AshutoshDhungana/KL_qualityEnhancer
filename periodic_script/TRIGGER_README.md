# Run the complete pipeline

python trigger.py

# Run for a specific product

python trigger.py --product Danfe_Corp_Product_2

# List all available steps

python trigger.py --list-steps

# Run only steps 2-4 (embedding, clustering, grouping)

python trigger.py --start-step 1 --end-step 4

# Preview what would be run without executing

python trigger.py --dry-run
