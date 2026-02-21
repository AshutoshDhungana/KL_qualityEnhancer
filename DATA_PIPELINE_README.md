# Data Merging Pipeline

This pipeline merges three data tables and organizes them by product, category, and data.

## Overview

The pipeline performs the following tasks:

1. Loads three CSV files from the data directory:
   - Product.csv: Contains product information
   - CanonicalQuestionProducts.csv: Maps questions to products
   - AnswerLibraryEntry.csv: Contains question answers and categories
2. Merges the tables on cqid and product_id using inner joins
3. Creates a directory structure organized by product > category > data

## Usage

1. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Run the pipeline:

   ```
   python src/Timed.py
   ```

3. The merged data will be saved in the `output` directory, organized by product and category.

## Data Structure

- Product.csv: Contains product_id, product_name, product_industry
- CanonicalQuestionProducts.csv: Contains cqid, product_id
- AnswerLibraryEntry.csv: Contains id, created_at, category, deleted_at, question, answer, details

## Output Structure

The pipeline creates the following directory structure:

```
output/
  ├── Product_Name_1/
  │   ├── Category_1/
  │   │   └── data.csv
  │   ├── Category_2/
  │   │   └── data.csv
  │   ...
  ├── Product_Name_2/
  ...
```
