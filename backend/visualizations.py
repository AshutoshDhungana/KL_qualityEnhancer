import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional

def generate_cluster_visualization(clusters_df: pd.DataFrame, method: str = 'umap') -> str:
    """
    Generate a cluster visualization using UMAP or t-SNE
    Returns a base64 encoded image
    """
    # For efficiency in the API, we'll use a simplified implementation
    # In a real application, this would use the actual embeddings data
    
    # Mock data for visualization
    n_clusters = len(clusters_df)
    n_points = clusters_df['size'].sum() if 'size' in clusters_df.columns else 500
    
    # Generate mock 2D coordinates
    mock_data = pd.DataFrame({
        'x': np.random.normal(0, 5, n_points),
        'y': np.random.normal(0, 5, n_points),
    })
    
    # Assign cluster labels
    cluster_labels = []
    for i, row in clusters_df.iterrows():
        size = row.get('size', 10)
        cluster_labels.extend([i] * size)
    
    # Ensure we have enough labels
    if len(cluster_labels) < n_points:
        cluster_labels.extend([n_clusters] * (n_points - len(cluster_labels)))
    mock_data['cluster'] = cluster_labels[:n_points]
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    for cluster_id in range(n_clusters):
        subset = mock_data[mock_data['cluster'] == cluster_id]
        plt.scatter(subset['x'], subset['y'], label=f'Cluster {cluster_id}', alpha=0.6)
    
    plt.title(f'Cluster Visualization ({method.upper()})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert to base64 for embedding in HTML
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f'data:image/png;base64,{img_str}'

def generate_similarity_heatmap(entries: List[Dict[str, Any]], similarity_matrix: Optional[np.ndarray] = None) -> str:
    """
    Generate a similarity heatmap for the given entries
    Returns a base64 encoded image
    """
    n_entries = len(entries)
    
    # If no similarity matrix provided, generate a mock one
    if similarity_matrix is None:
        # Create a mock similarity matrix (symmetric with 1s on diagonal)
        similarity_matrix = np.random.uniform(0.5, 0.95, (n_entries, n_entries))
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(similarity_matrix, 1.0)  # Diagonal is always 1
    
    # Create labels from entry questions or IDs
    labels = []
    for entry in entries:
        if 'question' in entry:
            # Truncate long questions
            q = entry['question']
            labels.append(q[:20] + '...' if len(q) > 20 else q)
        else:
            labels.append(str(entry.get('id', f'Entry {len(labels)+1}')))
    
    # Create heatmap
    fig = px.imshow(
        similarity_matrix,
        labels=dict(x="Entry", y="Entry", color="Similarity"),
        x=labels,
        y=labels,
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        title="Entry Similarity Heatmap",
        xaxis_title="",
        yaxis_title="",
        height=600
    )
    
    # Convert to HTML
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def generate_health_dashboard(summary_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate health dashboard visualizations
    Returns a dictionary with HTML for each plot
    """
    result = {}
    
    # Health score gauge
    if 'health_score' in summary_data:
        health_score = summary_data['health_score']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Health Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': health_score * 100
                }
            }
        ))
        
        result['health_gauge'] = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Cluster health distribution
    if 'cluster_health' in summary_data:
        health_data = summary_data['cluster_health']
        
        # Pie chart
        fig = px.pie(
            names=list(health_data.keys()),
            values=list(health_data.values()),
            title="Cluster Health Distribution",
            color=list(health_data.keys()),
            color_discrete_map={
                'Critical': '#F44336',
                'Needs Review': '#FF9800',
                'Healthy': '#4CAF50'
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        result['health_distribution'] = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    return result

def generate_clustering_metrics(cluster_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate metrics about clustering quality
    Returns a dictionary with HTML for each plot
    """
    result = {}
    
    # Cluster size distribution
    if 'cluster_sizes' in cluster_stats:
        cluster_sizes = cluster_stats['cluster_sizes']
        
        # Histogram of cluster sizes
        fig = px.histogram(
            x=cluster_sizes,
            nbins=20,
            title="Cluster Size Distribution",
            labels={'x': 'Cluster Size', 'y': 'Count'},
            color_discrete_sequence=['#1E88E5']
        )
        
        result['size_distribution'] = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Silhouette score over time
    if 'silhouette_scores' in cluster_stats:
        dates = cluster_stats.get('dates', list(range(len(cluster_stats['silhouette_scores']))))
        scores = cluster_stats['silhouette_scores']
        
        # Line chart
        fig = px.line(
            x=dates,
            y=scores,
            title="Clustering Quality Over Time (Silhouette Score)",
            labels={'x': 'Date', 'y': 'Silhouette Score'},
            color_discrete_sequence=['#1E88E5']
        )
        
        result['silhouette_trend'] = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    return result 