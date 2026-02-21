import os
import sys
import json
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path constants
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_CLUSTERS_PATH = BASE_DIR / "processed_clusters"
CLUSTERS_PATH = BASE_DIR / "clusters"
DATA_PATH = BASE_DIR / "data"
CLEANED_DATASET_PATH = BASE_DIR / "cleaned_dataset"
SCRIPT_PATH = BASE_DIR / "periodic_script"

# Import visualization functions
from visualizations import (
    generate_cluster_visualization,
    generate_similarity_heatmap,
    generate_health_dashboard,
    generate_clustering_metrics
)

# Import merge_utils
from merge_utils import QuestionMerger, merge_qa_pairs, log_merge_operation, merge_qa_pairs_by_ids, get_qa_pair_by_id

app = FastAPI(title="QnA Content Management API", 
              description="API for managing QnA content clusters and analysis")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Models --
class PipelineStep(BaseModel):
    name: str
    file: str
    args: List[str] = []
    
class PipelineOptions(BaseModel):
    product: Optional[str] = None
    start_step: int = 0
    end_step: Optional[int] = None
    dry_run: bool = False

class ClusterInfo(BaseModel):
    cluster_id: str
    size: int
    health_status: str
    similarity_score: float
    topics: List[str]
    
class PairwiseSimilarity(BaseModel):
    entry_id1: str
    entry_id2: str
    similarity_score: float

class QAPair(BaseModel):
    id: str
    question: str
    answer: str
    
class MergeRequest(BaseModel):
    pair1: QAPair
    pair2: QAPair
    user_id: str
    similarity_score: float = 0.0
    priority: Optional[str] = None  # 'pair1' or 'pair2' or None

class MergeByIdsRequest(BaseModel):
    cq_ids: List[str]
    user_id: str
    priority_id: Optional[str] = None  # ID of the question to prioritize
    similarity_score: float = 0.0

# -- API Routes --
@app.get("/")
async def root():
    return {"message": "QnA Content Management API"}

@app.get("/summary")
async def get_summary(product: Optional[str] = None):
    """Get overall summary statistics or product-specific summary if requested"""
    try:
        if product:
            # Format product name (replace spaces with underscores)
            product_name = product.replace(" ", "_")
            summary_file = PROCESSED_CLUSTERS_PATH / f"{product_name}_summary.json"
            
            # If product-specific summary doesn't exist, fall back to all_summary.json
            if not summary_file.exists():
                logger.warning(f"Product summary not found for {product_name}, using all_summary.json")
                summary_file = PROCESSED_CLUSTERS_PATH / "all_summary.json"
        else:
            summary_file = PROCESSED_CLUSTERS_PATH / "all_summary.json"
        
        # Check if the file exists
        if not summary_file.exists():
            raise HTTPException(status_code=404, detail=f"Summary file not found: {summary_file}")
        
        with open(summary_file, "r") as f:
            summary_data = json.load(f)
        
        # Clean the data to handle non-JSON-serializable values
        clean_summary = clean_json_data(summary_data)
        
        return clean_summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading summary data: {str(e)}")

@app.get("/products")
async def get_products():
    """Get list of available products"""
    try:
        products_df = pd.read_csv(DATA_PATH / "Product.csv")
        
        # Clean the dataframe to handle non-JSON-serializable values
        products_df_clean = clean_dataframe_for_json(products_df)
        
        return products_df_clean.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading products: {str(e)}")

@app.get("/clusters")
async def get_clusters(
    product: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    min_size: Optional[int] = None,
    health_status: Optional[str] = None
):
    """Get clusters with optional filtering"""
    try:
        # Determine which file to load based on product
        if product:
            file_path = PROCESSED_CLUSTERS_PATH / f"{product.replace(' ', '_')}_clusters.json"
            if not file_path.exists():
                # Fall back to all clusters and filter
                file_path = PROCESSED_CLUSTERS_PATH / "all_clusters.json"
        else:
            file_path = PROCESSED_CLUSTERS_PATH / "all_clusters.json"
            
        # Check if the file exists
        if not file_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Clusters file not found: {file_path}"
            )
        
        try:
            # Load the clusters from the JSON file
            with open(file_path, 'r') as f:
                clusters_data = json.load(f)
            
            # Extract the clusters array
            if 'clusters' in clusters_data:
                clusters_list = clusters_data['clusters']
            else:
                clusters_list = clusters_data  # Fallback if the data is directly an array
            
            # Convert to DataFrame for easier filtering
            clusters_df = pd.DataFrame(clusters_list)
            
            # Apply filters
            if product and "product" in clusters_df.columns:
                clusters_df = clusters_df[clusters_df["product"] == product]
                
            if min_size is not None and "size" in clusters_df.columns:
                clusters_df = clusters_df[clusters_df["size"] >= min_size]
                
            if health_status and "health_status" in clusters_df.columns:
                clusters_df = clusters_df[clusters_df["health_status"] == health_status]
            
            # Apply pagination
            paginated_df = clusters_df.iloc[offset:offset+limit]
            
            # Clean the dataframe to handle non-JSON-serializable values
            clean_df = clean_dataframe_for_json(paginated_df)
            
            # Convert to list of dictionaries
            result = clean_df.to_dict(orient="records")
            
            return result
        except ValueError as e:
            # Handle JSON parsing errors
            with open(file_path, 'r') as f:
                first_few_lines = f.read(1000)  # Read first 1000 chars to check format
            
            raise HTTPException(
                status_code=500,
                detail=f"Error parsing clusters file: {str(e)}. First few characters: {first_few_lines[:100]}..."
            )
                    
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error loading clusters: {str(e)}\nTrace: {trace}"
        )

@app.get("/clusters/{cluster_id}")
async def get_cluster_details(cluster_id: str):
    """Get detailed information about a specific cluster"""
    try:
        # Search in all cluster files for the specific cluster
        for file in PROCESSED_CLUSTERS_PATH.glob("*_clusters.json"):
            # Load clusters from JSON file
            with open(file, 'r') as f:
                clusters_data = json.load(f)
            
            # Extract the clusters array
            if 'clusters' in clusters_data:
                clusters_list = clusters_data['clusters']
            else:
                clusters_list = clusters_data
            
            # Convert to DataFrame
            clusters_df = pd.DataFrame(clusters_list)
            
            if "cluster_id" in clusters_df.columns:
                result = clusters_df[clusters_df["cluster_id"] == cluster_id]
                if not result.empty:
                    # Extract the first row as a Series
                    row = result.iloc[0]
                    
                    # Convert Series to DataFrame with a single row
                    single_row_df = pd.DataFrame([row])
                    
                    # Clean the dataframe to handle non-JSON-serializable values
                    clean_df = clean_dataframe_for_json(single_row_df)
                    
                    # Return the first (and only) row as a dict
                    return clean_df.iloc[0].to_dict()
        
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading cluster details: {str(e)}")

@app.get("/cluster/{cluster_id}/entries")
async def get_cluster_entries(cluster_id: str):
    """Get all QnA entries in a specific cluster"""
    try:
        # First find which product this cluster belongs to
        product = None
        
        for file in PROCESSED_CLUSTERS_PATH.glob("*_clusters.json"):
            # Load clusters from JSON file
            with open(file, 'r') as f:
                clusters_data = json.load(f)
            
            # Extract the clusters array
            if 'clusters' in clusters_data:
                clusters_list = clusters_data['clusters']
            else:
                clusters_list = clusters_data
            
            # Convert to DataFrame
            clusters_df = pd.DataFrame(clusters_list)
            
            if "cluster_id" in clusters_df.columns:
                result = clusters_df[clusters_df["cluster_id"] == cluster_id]
                if not result.empty and "product" in result.iloc[0]:
                    product = result.iloc[0]["product"]
                    break
        
        # Now load the relevant dataset with entries
        if product:
            dataset_path = CLEANED_DATASET_PATH / f"{product.replace(' ', '_')}_complete_dataset.csv"
            if not dataset_path.exists():
                dataset_path = CLEANED_DATASET_PATH / "all_complete_dataset.csv"
        else:
            dataset_path = CLEANED_DATASET_PATH / "all_complete_dataset.csv"
            
        # Load and filter the dataset
        dataset_df = pd.read_csv(dataset_path)
        if "cluster_id" in dataset_df.columns:
            result = dataset_df[dataset_df["cluster_id"] == cluster_id]
            
            # Clean the dataframe to handle non-JSON-serializable values
            result_clean = clean_dataframe_for_json(result)
            
            return result_clean.to_dict(orient="records")
        else:
            raise HTTPException(status_code=404, detail="Cluster entries not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading cluster entries: {str(e)}")

# Helper function to clean float values for JSON serialization
def clean_dataframe_for_json(df):
    """
    Clean a dataframe to ensure all values are JSON serializable
    Replaces NaN, Infinity, -Infinity with None/null
    """
    # Make a copy to avoid modifying the original dataframe
    df_clean = df.copy()
    
    # Replace NaN, Infinity, -Infinity with None/null
    for col in df_clean.columns:
        if df_clean[col].dtype.kind == 'f':  # Check if column is float type
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf, np.nan], None)
    
    return df_clean

# Helper function to clean dictionary/JSON data
def clean_json_data(data):
    """
    Clean a dictionary or JSON object to ensure all values are JSON serializable
    Replaces NaN, Infinity, -Infinity with None/null
    """
    if isinstance(data, dict):
        return {k: clean_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_data(item) for item in data]
    elif isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
        return None
    else:
        return data

@app.get("/outdated")
async def get_outdated_entries(
    limit: int = 100,
    offset: int = 0,
    product: Optional[str] = None,
    min_days: int = 180
):
    """Get potentially outdated entries based on last update date"""
    try:
        # Determine file path based on product
        if product:
            dataset_path = CLEANED_DATASET_PATH / f"{product.replace(' ', '_')}_complete_dataset.csv"
            if not dataset_path.exists():
                dataset_path = CLEANED_DATASET_PATH / "all_complete_dataset.csv"
        else:
            dataset_path = CLEANED_DATASET_PATH / "all_complete_dataset.csv"
            
        # Load the dataset
        dataset_df = pd.read_csv(dataset_path)
        
        # Filter by product if needed
        if product and "product" in dataset_df.columns:
            dataset_df = dataset_df[dataset_df["product"] == product]
            
        # Filter by update date if available
        if "last_updated" in dataset_df.columns:
            dataset_df["last_updated"] = pd.to_datetime(dataset_df["last_updated"])
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=min_days)
            dataset_df = dataset_df[dataset_df["last_updated"] < cutoff_date]
            
        # Apply pagination
        result = dataset_df.iloc[offset:offset+limit]
        
        # Clean the dataframe to handle non-JSON-serializable values
        result_clean = clean_dataframe_for_json(result)
        
        return result_clean.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading outdated entries: {str(e)}")

@app.get("/pipeline/steps")
async def get_pipeline_steps():
    """Get available pipeline steps"""
    steps = [
        {"name": "Timed", "file": "timed.py", "args": []},
        {"name": "Embedding", "file": "embedding.py", "args": []},
        {"name": "Clustering", "file": "clustering.py", "args": []},
        {"name": "Cluster Grouping", "file": "cluster_grouping.py", "args": []},
        {"name": "Cleaned Dataset Creation", "file": "cleaned_ds_create.py", "args": []},
        {"name": "Cluster Cache Generation", "file": "cluster_cache.py", "args": []}
    ]
    return steps

def run_pipeline_task(options: PipelineOptions):
    """Background task to run the pipeline"""
    cmd = [sys.executable, str(SCRIPT_PATH / "trigger.py")]
    
    if options.product:
        cmd.extend(["--product", options.product])
        
    if options.start_step is not None:
        cmd.extend(["--start-step", str(options.start_step)])
        
    if options.end_step is not None:
        cmd.extend(["--end-step", str(options.end_step)])
        
    if options.dry_run:
        cmd.append("--dry-run")
        
    # Run the command
    subprocess.run(cmd, check=True)

@app.post("/pipeline/run")
async def run_pipeline(options: PipelineOptions, background_tasks: BackgroundTasks):
    """Run the pipeline with specified options"""
    try:
        # Start the pipeline in the background
        background_tasks.add_task(run_pipeline_task, options)
        return {"message": "Pipeline started", "options": options.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting pipeline: {str(e)}")

@app.get("/similar-pairs")
async def get_similar_pairs(
    min_similarity: float = 0.8,
    limit: int = 100,
    offset: int = 0,
    product: Optional[str] = None
):
    """Get pairs of entries with high similarity scores"""
    try:
        # This would typically load from a precomputed similarity table
        # For now, using a placeholder implementation
        if product:
            similar_pairs_path = CLUSTERS_PATH / f"{product.replace(' ', '_')}_similar_pairs.csv"
            if not similar_pairs_path.exists():
                similar_pairs_path = CLUSTERS_PATH / "all_similar_pairs.csv"
        else:
            similar_pairs_path = CLUSTERS_PATH / "all_similar_pairs.csv"
            
        # Check if the file exists, if not return empty result
        if not similar_pairs_path.exists():
            return []
            
        # Load the similarity pairs data
        pairs_df = pd.read_csv(similar_pairs_path)
        
        # Filter by similarity threshold
        if "similarity_score" in pairs_df.columns:
            pairs_df = pairs_df[pairs_df["similarity_score"] >= min_similarity]
            
        # Apply pagination
        result = pairs_df.iloc[offset:offset+limit]
        
        # Clean the dataframe to handle non-JSON-serializable values
        result_clean = clean_dataframe_for_json(result)
        
        return result_clean.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading similarity pairs: {str(e)}")

# -- Visualization Endpoints --
@app.get("/visualizations/clusters", response_class=HTMLResponse)
async def get_cluster_visualization(
    product: Optional[str] = None,
    method: str = "umap"
):
    """Get cluster visualization"""
    try:
        # Load clusters data
        if product:
            file_path = PROCESSED_CLUSTERS_PATH / f"{product.replace(' ', '_')}_clusters.json"
            if not file_path.exists():
                file_path = PROCESSED_CLUSTERS_PATH / "all_clusters.json"
        else:
            file_path = PROCESSED_CLUSTERS_PATH / "all_clusters.json"
            
        # Check if the file exists
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Clusters file not found")
            
        # Load clusters data
        clusters_df = pd.read_json(file_path, lines=True)
        
        # Generate visualization
        img_data = generate_cluster_visualization(clusters_df, method)
        
        # Return HTML with embedded image
        html_content = f"""
        <html>
            <head>
                <title>Cluster Visualization</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                    h1 {{ color: #1E88E5; }}
                    img {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
                <h1>Cluster Visualization ({method.upper()})</h1>
                <img src="{img_data}" alt="Cluster Visualization">
            </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")

@app.get("/visualizations/similarity/{cluster_id}", response_class=HTMLResponse)
async def get_similarity_visualization(cluster_id: str):
    """Get similarity visualization for entries in a cluster"""
    try:
        # Get entries in the cluster
        entries = await get_cluster_entries(cluster_id)
        
        if not entries:
            raise HTTPException(status_code=404, detail=f"No entries found for cluster {cluster_id}")
            
        # Generate similarity heatmap
        heatmap_html = generate_similarity_heatmap(entries)
        
        # Return HTML with embedded visualization
        html_content = f"""
        <html>
            <head>
                <title>Similarity Visualization for Cluster {cluster_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                    h1 {{ color: #1E88E5; }}
                </style>
            </head>
            <body>
                <h1>Similarity Visualization for Cluster {cluster_id}</h1>
                {heatmap_html}
            </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating similarity visualization: {str(e)}")

@app.get("/visualizations/health-dashboard", response_class=HTMLResponse)
async def get_health_dashboard():
    """Get health dashboard visualizations"""
    try:
        # Get summary data
        summary_data = await get_summary()
        
        if not summary_data:
            raise HTTPException(status_code=404, detail="Summary data not found")
            
        # Generate health dashboard visualizations
        dashboard = generate_health_dashboard(summary_data)
        
        # Clean any potential non-serializable data
        dashboard = clean_json_data(dashboard)
        
        # Return HTML with embedded visualizations
        html_content = f"""
        <html>
            <head>
                <title>Health Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                    h1, h2 {{ color: #1E88E5; }}
                    .dashboard-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                    .dashboard-item {{ flex: 1; min-width: 400px; }}
                </style>
            </head>
            <body>
                <h1>Health Dashboard</h1>
                <div class="dashboard-container">
        """
        
        for key, visualization in dashboard.items():
            html_content += f"""
                    <div class="dashboard-item">
                        <h2>{key.replace('_', ' ').title()}</h2>
                        {visualization}
                    </div>
            """
            
        html_content += """
                </div>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating health dashboard: {str(e)}")

@app.get("/visualizations/cluster-metrics", response_class=HTMLResponse)
async def get_cluster_metrics(product: Optional[str] = None):
    """Get clustering metrics visualizations"""
    try:
        # Mock cluster stats for demonstration
        # In a real implementation, this would load from actual data
        cluster_stats = {
            "cluster_sizes": [5, 8, 12, 15, 7, 10, 25, 18, 6, 9, 11, 14, 22, 30, 5, 8, 10],
            "silhouette_scores": [0.65, 0.68, 0.72, 0.70, 0.67],
            "dates": ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05"]
        }
        
        # Clean the stats to ensure they're JSON serializable
        cluster_stats = clean_json_data(cluster_stats)
        
        # Generate clustering metrics visualizations
        metrics = generate_clustering_metrics(cluster_stats)
        
        # Clean any potential non-serializable data in the result
        metrics = clean_json_data(metrics)
        
        # Return HTML with embedded visualizations
        html_content = f"""
        <html>
            <head>
                <title>Clustering Metrics</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                    h1, h2 {{ color: #1E88E5; }}
                    .metrics-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                    .metric-item {{ flex: 1; min-width: 400px; }}
                </style>
            </head>
            <body>
                <h1>Clustering Metrics {f'for {product}' if product else ''}</h1>
                <div class="metrics-container">
        """
        
        for key, visualization in metrics.items():
            html_content += f"""
                    <div class="metric-item">
                        <h2>{key.replace('_', ' ').title()}</h2>
                        {visualization}
                    </div>
            """
            
        html_content += """
                </div>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating clustering metrics: {str(e)}")

@app.post("/merge-qa-pairs")
async def api_merge_qa_pairs(merge_request: MergeRequest):
    """
    Merge two QA pairs using the merge_utils pipeline
    
    Args:
        merge_request: MergeRequest object containing the two QA pairs to merge
        
    Returns:
        The merged QA pair
    """
    try:
        # Merge the QA pairs with priority if specified
        merged_pair = merge_qa_pairs(
            merge_request.pair1.dict(), 
            merge_request.pair2.dict(),
            priority=merge_request.priority
        )
        
        # Generate a new ID for the merged pair
        merged_pair["id"] = f"merged_{merge_request.pair1.id}_{merge_request.pair2.id}"
        
        # Log the merge operation
        log_merge_operation(
            user_id=merge_request.user_id,
            pair1_id=merge_request.pair1.id,
            pair2_id=merge_request.pair2.id,
            result_id=merged_pair["id"],
            similarity_score=merge_request.similarity_score,
            priority=merge_request.priority
        )
        
        # Return the merged pair
        return merged_pair
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error merging QA pairs: {str(e)}"
        )

@app.post("/merge-qa-pairs-by-ids")
async def api_merge_qa_pairs_by_ids(merge_request: MergeByIdsRequest):
    """
    Merge multiple QA pairs by their IDs using the merge_utils pipeline
    
    Args:
        merge_request: MergeByIdsRequest object containing the IDs of QA pairs to merge
        
    Returns:
        The merged QA pair
    """
    try:
        # Validate that we have at least two IDs
        if len(merge_request.cq_ids) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least two question IDs are required for merging"
            )
        
        # Merge the QA pairs by their IDs with priority if specified
        merged_pair = merge_qa_pairs_by_ids(
            merge_request.cq_ids,
            priority_id=merge_request.priority_id
        )
        
        if not merged_pair:
            raise HTTPException(
                status_code=404,
                detail="One or more QA pairs could not be found with the provided IDs"
            )
        
        # Log the merge operation
        log_merge_operation(
            user_id=merge_request.user_id,
            pair1_id=merge_request.cq_ids[0],
            pair2_id=merge_request.cq_ids[1] if len(merge_request.cq_ids) > 1 else "multiple",
            result_id=merged_pair["id"],
            similarity_score=merge_request.similarity_score,
            priority=merge_request.priority_id
        )
        
        # Return the merged pair
        return merged_pair
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error merging QA pairs by IDs: {str(e)}"
        )

@app.post("/save-merged-pair")
async def save_merged_qa_pair(merged_pair: QAPair):
    """
    Save a merged QA pair to a file
    
    Args:
        merged_pair: QAPair object containing the merged question and answer
        
    Returns:
        The file path where the merged pair was saved
    """
    try:
        # Initialize the merger utility
        merger = QuestionMerger()
        
        # Create a directory for merged pairs if it doesn't exist
        output_dir = BASE_DIR / "merged_qa_pairs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a file name for the merged pair
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"merged_qa_{timestamp}_{merged_pair.id}.json"
        output_file = output_dir / file_name
        
        # Save the merged pair to a file
        pair_dict = merged_pair.dict()
        pair_dict["merged_at"] = datetime.now().isoformat()
        pair_dict["is_canonical"] = True
        
        success = merger.save_merged_pair(pair_dict, str(output_file))
        
        if success:
            return {
                "success": True,
                "file_path": str(output_file),
                "merged_pair": pair_dict
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to save merged pair"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving merged QA pair: {str(e)}"
        )

@app.get("/merged-qa-pairs")
async def get_merged_qa_pairs():
    """
    Get a list of all saved merged QA pairs
    
    Returns:
        A list of merged QA pairs
    """
    try:
        # Create a directory for merged pairs if it doesn't exist
        output_dir = BASE_DIR / "merged_qa_pairs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get a list of all JSON files in the directory
        merged_files = list(output_dir.glob("*.json"))
        
        # Read each file and collect the merged pairs
        merged_pairs = []
        for file_path in merged_files:
            try:
                with open(file_path, "r") as f:
                    merged_pair = json.load(f)
                    merged_pair["file_path"] = str(file_path)
                    merged_pairs.append(merged_pair)
            except Exception as e:
                # Skip files that can't be read
                continue
        
        # Sort by merged_at timestamp, newest first
        merged_pairs.sort(key=lambda x: x.get("merged_at", ""), reverse=True)
        
        return merged_pairs
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing merged QA pairs: {str(e)}"
        ) 