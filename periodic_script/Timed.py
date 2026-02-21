import pandas as pd
import os
from pathlib import Path

def load_data(data_dir=None):
    """
    Load the three CSV files from the data directory
    """
    # Use absolute path if data_dir is not provided
    if data_dir is None:
        # Get the parent directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        data_dir = os.path.join(base_dir, "data")
    
    print(f"Loading data from: {data_dir}")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Load the Product table
    product_file = os.path.join(data_dir, "Product.csv")
    if not os.path.exists(product_file):
        raise FileNotFoundError(f"Product file not found: {product_file}")
    product_df = pd.read_csv(product_file)
    print(f"Loaded Product data: {product_df.shape[0]} rows")
    
    # Load the Canonical Questions Products table - it has no header
    cqp_file = os.path.join(data_dir, "CanonicalQuestionProducts.csv")
    if not os.path.exists(cqp_file):
        raise FileNotFoundError(f"Canonical Questions Products file not found: {cqp_file}")
    cqp_df = pd.read_csv(cqp_file, header=None, names=["cqid", "product_id"])
    print(f"Loaded Canonical Questions Products data: {cqp_df.shape[0]} rows")
    
    # Load the Answer table
    answer_file = os.path.join(data_dir, "AnswerLibraryEntry.csv")
    if not os.path.exists(answer_file):
        raise FileNotFoundError(f"Answer Library Entry file not found: {answer_file}")
    answer_df = pd.read_csv(answer_file)
    print(f"Loaded Answer data: {answer_df.shape[0]} rows")
    
    return product_df, cqp_df, answer_df

def merge_tables(product_df, cqp_df, answer_df):
    """
    Merge the three tables using inner join on product_id and cqid
    """
    # First merge product and canonical questions
    merged_df = pd.merge(cqp_df, product_df, on="product_id", how="inner")
    print(f"After merging product data: {merged_df.shape[0]} rows")
    
    # Then merge with answers (cqid from CQP table matches id from Answer table)
    full_merged_df = pd.merge(merged_df, answer_df, left_on="cqid", right_on="id", how="inner")
    print(f"After merging answer data: {full_merged_df.shape[0]} rows")
    
    return full_merged_df

def create_product_category_tables(merged_df, output_dir=None):
    """
    Create separate tables organized by product > category > data
    """
    # Use absolute path if output_dir is not provided
    if output_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        output_dir = os.path.join(base_dir, "output")
    
    print(f"Saving output to: {output_dir}")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Group by product and category
    for product_id, product_group in merged_df.groupby("product_id"):
        product_name = product_group["product_name"].iloc[0]
        product_dir = os.path.join(output_dir, product_name.replace(' ', '_'))
        Path(product_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Processing product: {product_name}")
        
        # Group by category within each product
        for category, category_group in product_group.groupby("category"):
            if pd.isna(category):
                category = "uncategorized"
            else:
                # Ensure category is a valid directory name
                category = str(category).replace(' ', '_').replace('/', '_').replace('\\', '_')
            
            category_dir = os.path.join(product_dir, category)
            Path(category_dir).mkdir(parents=True, exist_ok=True)
            
            # Save the category data
            file_path = os.path.join(category_dir, "data.csv")
            category_group.reset_index(drop=True).to_csv(file_path, index=False)
            print(f"  - Saved {category} data: {category_group.shape[0]} rows to {file_path}")

def main():
    """
    Main function to run the pipeline
    """
    print("Starting data pipeline...")
    
    # Load data
    product_df, cqp_df, answer_df = load_data()
    
    # Merge tables
    merged_df = merge_tables(product_df, cqp_df, answer_df)
    
    # Create product > category > data tables
    create_product_category_tables(merged_df)
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
