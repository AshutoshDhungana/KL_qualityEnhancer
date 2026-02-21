import os
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import json
import time
import logging
from tqdm import tqdm
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("embedding_pipeline.log"), logging.StreamHandler()]
)
logger = logging.getLogger("embedding_pipeline")

def normalize_path(path):
    """Helper function to normalize and make path absolute"""
    if path.startswith(".."):
        # Handle relative paths like ../output
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_dir, path))
    elif os.path.isabs(path):
        return path
    else:
        # Handle relative paths without ../
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        return os.path.join(base_dir, path)

class EmbeddingPipeline:
    def __init__(self, 
                 output_dir="output", 
                 model_name="all-MiniLM-L6-v2", 
                 models_dir="models"):
        """
        Initialize the embedding pipeline.
        
        Args:
            output_dir: Directory containing the output data organized by product > category
            model_name: Name of the sentence transformer model to use
            models_dir: Directory to save embeddings and indices
        """
        self.output_dir = output_dir
        self.model_name = model_name
        self.models_dir = models_dir
        self.model = None
        
        # Normalize paths
        self.abs_output_dir = normalize_path(self.output_dir)
        self.abs_models_dir = normalize_path(self.models_dir)
        
        logger.info(f"Output directory: {self.abs_output_dir}")
        logger.info(f"Models directory: {self.abs_models_dir}")
        
        # Create models directory if it doesn't exist
        Path(self.abs_models_dir).mkdir(parents=True, exist_ok=True)
        
        # Structure to store embeddings
        self.embeddings_data = {
            "products": {}
        }
        
        logger.info(f"Initialized embedding pipeline with model: {model_name}")
    
    def get_absolute_path(self, *paths):
        """Helper method to get absolute paths"""
        # Just join the paths - we're now using normalized absolute paths throughout
        return os.path.join(*paths)
    
    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            logger.info(f"Loading model: {self.model_name}")
            start_time = time.time()
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        return self.model
    
    def get_products(self):
        """Get list of product directories in the output folder"""
        logger.info(f"Looking for products in: {self.abs_output_dir}")
        
        # Check if the directory exists
        if not os.path.exists(self.abs_output_dir):
            logger.error(f"Output directory not found: {self.abs_output_dir}")
            return []
        
        # Get list of product directories
        try:
            products = [d for d in os.listdir(self.abs_output_dir) 
                      if os.path.isdir(os.path.join(self.abs_output_dir, d))]
            logger.info(f"Found {len(products)} products: {', '.join(products)}")
            return products
        except Exception as e:
            logger.error(f"Error getting products: {str(e)}")
            return []
    
    def get_categories(self, product):
        """Get list of category directories for a product"""
        product_dir = os.path.join(self.abs_output_dir, product)
        
        logger.info(f"Looking for categories in product: {product_dir}")
        
        # Check if the directory exists
        if not os.path.exists(product_dir):
            logger.error(f"Product directory not found: {product_dir}")
            return []
        
        # Get list of category directories
        try:
            categories = [d for d in os.listdir(product_dir) 
                        if os.path.isdir(os.path.join(product_dir, d))]
            logger.info(f"Found {len(categories)} categories for product {product}")
            return categories
        except Exception as e:
            logger.error(f"Error getting categories: {str(e)}")
            return []
    
    def load_category_data(self, product, category):
        """Load data for a specific product and category"""
        data_path = os.path.join(self.abs_output_dir, product, category, "data.csv")
        
        logger.info(f"Loading data from: {data_path}")
        
        # Check if the file exists
        if not os.path.exists(data_path):
            logger.warning(f"Data file not found: {data_path}")
            return None
        
        # Load the data
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} rows from {data_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
    
    def create_embeddings(self):
        """Create embeddings for all products, categories, and questions"""
        # Load the model
        self.load_model()
        
        # Process each product
        products = self.get_products()
        logger.info(f"Processing {len(products)} products")
        
        for product in products:
            logger.info(f"Processing product: {product}")
            self.embeddings_data["products"][product] = {"categories": {}}
            
            # Create product directory in models_dir if it doesn't exist
            product_models_dir = os.path.join(self.abs_models_dir, product)
            Path(product_models_dir).mkdir(parents=True, exist_ok=True)
            
            # Process each category
            categories = self.get_categories(product)
            logger.info(f"Processing {len(categories)} categories for product: {product}")
            
            for category in tqdm(categories, desc=f"Categories in {product}"):
                # Load category data
                df = self.load_category_data(product, category)
                if df is None or df.empty:
                    continue
                
                # Check if 'question' column exists
                if 'question' not in df.columns:
                    logger.warning(f"No 'question' column in {product}/{category}")
                    continue
                
                # Get questions and their IDs
                questions = df['question'].tolist()
                question_ids = df['id'].tolist()
                
                # Generate embeddings for questions
                logger.info(f"Generating embeddings for {len(questions)} questions in {product}/{category}")
                question_embeddings = self.model.encode(questions, show_progress_bar=True)
                
                # Create a FAISS index for this category
                embedding_dim = question_embeddings.shape[1]
                index = faiss.IndexFlatL2(embedding_dim)
                index.add(np.array(question_embeddings).astype('float32'))
                
                # Save category embeddings data
                category_data = {
                    "question_ids": question_ids,
                    "questions": questions,
                    "embeddings_file": f"{category}_embeddings.npy",
                    "index_file": f"{category}_index.faiss"
                }
                
                self.embeddings_data["products"][product]["categories"][category] = category_data
                
                # Save embeddings and index to product-specific directory
                embeddings_path = os.path.join(product_models_dir, category_data["embeddings_file"])
                index_path = os.path.join(product_models_dir, category_data["index_file"])
                
                np.save(embeddings_path, question_embeddings)
                faiss.write_index(index, index_path)
                
                logger.info(f"Saved embeddings and index for {product}/{category}")
            
            # Save the product-specific metadata
            product_metadata_path = os.path.join(product_models_dir, "product_metadata.json")
            with open(product_metadata_path, 'w') as f:
                json.dump(self.embeddings_data["products"][product], f, indent=2)
        
        # Save the main metadata
        metadata_path = os.path.join(self.abs_models_dir, "embeddings_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.embeddings_data, f, indent=2)
        
        logger.info(f"Embedding pipeline completed. Metadata saved to: {metadata_path}")
        
        return self.embeddings_data
    
    def search_similar_questions(self, query, product=None, category=None, top_k=5):
        """
        Search for similar questions across all products and categories or in a specific product/category.
        
        Args:
            query: The question to search for
            product: (Optional) Specific product to search in
            category: (Optional) Specific category to search in
            top_k: Number of results to return
            
        Returns:
            List of tuples (product, category, question, similarity_score)
        """
        # Load the model
        self.load_model()
        
        # Generate embedding for the query
        query_embedding = self.model.encode([query])[0]
        
        # Load metadata
        metadata_path = os.path.join(self.abs_models_dir, "embeddings_metadata.json")
        if not os.path.exists(metadata_path):
            logger.error("Metadata file not found. Run create_embeddings() first.")
            return []
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        results = []
        products_to_search = [product] if product else list(metadata["products"].keys())
        
        for p in products_to_search:
            if p not in metadata["products"]:
                logger.warning(f"Product {p} not found in metadata")
                continue
            
            # Get the product-specific models directory
            product_models_dir = os.path.join(self.abs_models_dir, p)
            if not os.path.exists(product_models_dir):
                logger.warning(f"Product models directory not found: {product_models_dir}")
                continue
            
            categories_to_search = [category] if category else list(metadata["products"][p]["categories"].keys())
            
            for c in categories_to_search:
                if c not in metadata["products"][p]["categories"]:
                    logger.warning(f"Category {c} not found in product {p}")
                    continue
                
                category_data = metadata["products"][p]["categories"][c]
                
                # Load embeddings and index from product folder
                embeddings_path = os.path.join(product_models_dir, category_data["embeddings_file"])
                index_path = os.path.join(product_models_dir, category_data["index_file"])
                
                if not os.path.exists(embeddings_path) or not os.path.exists(index_path):
                    logger.warning(f"Embeddings or index not found for {p}/{c}")
                    continue
                
                # Load the index
                index = faiss.read_index(index_path)
                
                # Search
                scores, indices = index.search(np.array([query_embedding]).astype('float32'), top_k)
                
                # Add results
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx < len(category_data["questions"]):  # Ensure index is valid
                        # Convert distance to similarity score (1 - normalized_distance)
                        similarity = 1 - (score / 100)  # Normalize the distance
                        results.append((
                            p,  # product
                            c,  # category
                            category_data["questions"][idx],  # question
                            category_data["question_ids"][idx],  # question_id
                            similarity  # similarity score
                        ))
        
        # Sort results by similarity (highest first)
        results.sort(key=lambda x: x[4], reverse=True)
        
        # Return top results
        return results[:top_k]


def main():
    """
    Main function to run the embedding pipeline
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Embedding pipeline for product-category-question data")
    parser.add_argument("--output-dir", default="output", help="Directory containing organized data (default: output)")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2", help="Name of the sentence transformer model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--models-dir", default="models", help="Directory to save embeddings and indices (default: models)")
    parser.add_argument("--query", default=None, help="Test query to search for (optional)")
    args = parser.parse_args()
    
    logger.info("Starting embedding pipeline...")
    
    try:
        # Normalize paths from command line arguments
        output_dir = normalize_path(args.output_dir)
        models_dir = normalize_path(args.models_dir)
        
        logger.info(f"Using output directory: {output_dir}")
        logger.info(f"Using models directory: {models_dir}")
        
        # Check if output directory exists
        if not os.path.exists(output_dir):
            logger.error(f"Output directory not found: {output_dir}")
            return 1
            
        # Initialize pipeline with absolute paths
        pipeline = EmbeddingPipeline(
            output_dir=output_dir,
            model_name=args.model_name,
            models_dir=models_dir
        )
        
        # Create embeddings
        pipeline.create_embeddings()
        
        # Example search if a query was provided
        if args.query:
            logger.info(f"Testing search functionality with query: '{args.query}'")
            results = pipeline.search_similar_questions(args.query, top_k=5)
            
            logger.info(f"Top 5 results for query: '{args.query}'")
            for product, category, question, question_id, score in results:
                logger.info(f"Product: {product}, Category: {category}, Question: '{question}', Score: {score:.4f}")
        
        logger.info("Embedding pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Error running embedding pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
