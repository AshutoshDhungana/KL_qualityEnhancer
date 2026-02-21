import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import requests
import os
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuration
API_URL = "http://localhost:8000"  # FastAPI backend URL

# Set page configuration
st.set_page_config(
    page_title="QnA Content Management Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
        margin-bottom: 0.5rem;
    }
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        text-align: center;
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #ffffff;
        box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #757575;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-healthy {
        background-color: #4CAF50;
        color: white;
    }
    .status-review {
        background-color: #FF9800;
        color: white;
    }
    .status-critical {
        background-color: #F44336;
        color: white;
    }
    .sidebar-section {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache API calls
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_api_data(endpoint, params=None):
    try:
        response = requests.get(f"{API_URL}{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data from API: {str(e)}")
        return None

def run_pipeline(options):
    try:
        response = requests.post(f"{API_URL}/pipeline/run", json=options)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error triggering pipeline: {str(e)}")
        return None

# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/question-mark.png", width=60)
    st.title("QnA Management")
    
    st.markdown("### Navigation")
    page = st.radio(
        "Select Page",
        ["Dashboard", "Cluster Explorer", "Similarity Analysis", 
         "Outdated Content", "Review Panel", "Pipeline Control"]
    )
    
    st.markdown("---")
    
    st.markdown("### Filters")
    # Load products for filtering
    products = fetch_api_data("/products")
    if products:
        product_options = ["All Products"] + [p["product_name"] for p in products]
        selected_product = st.selectbox("Product", product_options)
        product_filter = None if selected_product == "All Products" else selected_product
    else:
        product_filter = None
    
    st.markdown("---")
    
    # Footer
    st.markdown("*Dashboard v1.0*")
    st.markdown("*© 2023 SecurityPal*")

# Main content area
if page == "Dashboard":
    st.markdown("<h1 class='main-header'>QnA Content Management Dashboard</h1>", unsafe_allow_html=True)
    
    # Get summary data
    summary_data = fetch_api_data("/summary")
    
    if summary_data:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{summary_data.get('total_questions', 0):,}</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Total Questions</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{summary_data.get('clusters', {}).get('total', 0):,}</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Total Clusters</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            # Calculate average cluster size from the cluster_size_distribution if not directly provided
            avg_size = summary_data.get('avg_cluster_size', 0)
            if avg_size == 0 and 'cluster_size_distribution' in summary_data:
                if 'mean' in summary_data['cluster_size_distribution']:
                    avg_size = summary_data['cluster_size_distribution']['mean']
            
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{avg_size:.1f}</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Avg. Cluster Size</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col4:
            # If health_score doesn't exist, calculate a simple one
            health_score = summary_data.get('health_score', 0)
            if health_score == 0 and 'questions' in summary_data:
                # Simple health score based on canonical questions ratio
                total = summary_data['total_questions']
                canonical = summary_data.get('questions', {}).get('canonical', 0)
                if total > 0:
                    health_score = canonical / total
            
            health_pct = health_score * 100
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{health_pct:.1f}%</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Health Score</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Cluster health distribution
        st.markdown("<h2 class='sub-header'>Cluster Health Distribution</h2>", unsafe_allow_html=True)
        
        # Create default health distribution if it doesn't exist
        if 'cluster_health' not in summary_data:
            health_data = {
                'Healthy': summary_data.get('clusters', {}).get('total', 0) // 3,
                'Needs Review': summary_data.get('clusters', {}).get('total', 0) // 3,
                'Critical': summary_data.get('clusters', {}).get('total', 0) // 3
            }
        else:
            health_data = summary_data['cluster_health']
        
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
        st.plotly_chart(fig, use_container_width=True)
        
        # Products table
        st.markdown("<h2 class='sub-header'>Products</h2>", unsafe_allow_html=True)
        if products:
            # Convert to DataFrame for display
            products_df = pd.DataFrame(products)
            st.dataframe(products_df, use_container_width=True)
    else:
        st.warning("Unable to load summary data. Please ensure the backend API is running.")

elif page == "Cluster Explorer":
    st.markdown("<h1 class='main-header'>Cluster Explorer</h1>", unsafe_allow_html=True)
    
    # Parameters for cluster filtering
    col1, col2, col3 = st.columns(3)
    with col1:
        min_size = st.number_input("Min Cluster Size", min_value=2, value=5)
    with col2:
        health_status_options = ["All", "Healthy", "Needs Review", "Critical"]
        health_status = st.selectbox("Health Status", health_status_options)
        health_filter = None if health_status == "All" else health_status
    with col3:
        limit = st.slider("Max Clusters to Show", min_value=10, max_value=100, value=50)
    
    # Get clusters based on filters
    params = {
        "limit": limit,
        "offset": 0,
        "min_size": min_size
    }
    if product_filter:
        params["product"] = product_filter
    if health_filter:
        params["health_status"] = health_filter
    
    clusters = fetch_api_data("/clusters", params)
    
    if clusters:
        st.markdown(f"<h2 class='sub-header'>Found {len(clusters)} clusters</h2>", unsafe_allow_html=True)
        
        # Display clusters
        for i, cluster in enumerate(clusters):
            with st.expander(f"Cluster {i+1}: {cluster.get('cluster_id', 'Unknown')} - Size: {cluster.get('size', 0)}"):
                # First column for cluster details
                st.markdown("### Cluster Details")
                
                # Format health status with badge
                health = cluster.get('health_status', 'Unknown')
                health_class = {
                    'Healthy': 'status-healthy',
                    'Needs Review': 'status-review',
                    'Critical': 'status-critical'
                }.get(health, '')
                
                st.markdown(f"""
                * **ID**: {cluster.get('cluster_id', 'Unknown')}
                * **Size**: {cluster.get('size', 0)} entries
                * **Health**: <span class='status-badge {health_class}'>{health}</span>
                * **Similarity Score**: {cluster.get('similarity_score', 0):.2f}
                * **Product**: {cluster.get('product', 'Unknown')}
                """, unsafe_allow_html=True)
                
                if 'topics' in cluster and cluster['topics']:
                    st.markdown("#### Key Topics")
                    for topic in cluster['topics']:
                        st.markdown(f"* {topic}")
                
                # Load entries in this cluster
                if st.button(f"View Entries in Cluster {i+1}"):
                    cluster_id = cluster.get('cluster_id')
                    entries = fetch_api_data(f"/cluster/{cluster_id}/entries")
                    
                    if entries:
                        st.markdown("### Entries in Cluster")
                        entries_df = pd.DataFrame(entries)
                        
                        # Select the most important columns to display
                        display_cols = ['question', 'answer', 'last_updated'] if 'question' in entries_df.columns else entries_df.columns
                        st.dataframe(entries_df[display_cols], use_container_width=True)
                    else:
                        st.warning("No entries found for this cluster.")
    else:
        st.warning("No clusters found matching the criteria.")

elif page == "Similarity Analysis":
    st.markdown("<h1 class='main-header'>Similarity Analysis</h1>", unsafe_allow_html=True)
    
    # Parameters for similarity analysis
    col1, col2 = st.columns(2)
    with col1:
        min_similarity = st.slider("Minimum Similarity Score", 0.7, 1.0, 0.9, 0.01)
    with col2:
        limit = st.slider("Max Pairs to Show", min_value=10, max_value=100, value=25)
    
    # Get similar pairs
    params = {
        "min_similarity": min_similarity,
        "limit": limit,
        "offset": 0
    }
    if product_filter:
        params["product"] = product_filter
    
    similar_pairs = fetch_api_data("/similar-pairs", params)
    
    if similar_pairs:
        st.markdown(f"<h2 class='sub-header'>Found {len(similar_pairs)} high-similarity pairs</h2>", unsafe_allow_html=True)
        
        # Display similarity pairs
        for i, pair in enumerate(similar_pairs):
            with st.expander(f"Pair {i+1}: Similarity Score {pair.get('similarity_score', 0):.2f}"):
                col1, col2 = st.columns(2)
                
                entry1_id = pair.get('entry_id1', '')
                entry2_id = pair.get('entry_id2', '')
                
                # In a real implementation, we would fetch the actual entry details
                # For now, just showing the IDs
                with col1:
                    st.markdown(f"### Entry 1 (ID: {entry1_id})")
                    st.markdown("Question and answer would appear here...")
                
                with col2:
                    st.markdown(f"### Entry 2 (ID: {entry2_id})")
                    st.markdown("Question and answer would appear here...")
                
                # Action buttons
                st.markdown("---")
                action_col1, action_col2, action_col3, action_col4 = st.columns(4)
                
                with action_col1:
                    if st.button(f"Merge Entries {i+1}", key=f"merge_{i}"):
                        st.success("Entries would be merged (action not implemented)")
                
                with action_col2:
                    if st.button(f"Mark as Different {i+1}", key=f"diff_{i}"):
                        st.info("Entries marked as different (action not implemented)")
                
                with action_col3:
                    if st.button(f"Flag for Review {i+1}", key=f"flag_{i}"):
                        st.warning("Entries flagged for review (action not implemented)")
                
                with action_col4:
                    if st.button(f"Reject Suggestion {i+1}", key=f"reject_{i}"):
                        st.error("Suggestion rejected (action not implemented)")
    else:
        st.warning("No high-similarity pairs found matching the criteria.")

elif page == "Outdated Content":
    st.markdown("<h1 class='main-header'>Outdated Content</h1>", unsafe_allow_html=True)
    
    # Parameters for outdated content
    col1, col2 = st.columns(2)
    with col1:
        min_days = st.slider("Minimum Days Since Update", 30, 365, 180)
    with col2:
        limit = st.slider("Max Entries to Show", min_value=10, max_value=100, value=50)
    
    # Get outdated entries
    params = {
        "min_days": min_days,
        "limit": limit,
        "offset": 0
    }
    if product_filter:
        params["product"] = product_filter
    
    outdated_entries = fetch_api_data("/outdated", params)
    
    if outdated_entries:
        st.markdown(f"<h2 class='sub-header'>Found {len(outdated_entries)} outdated entries</h2>", unsafe_allow_html=True)
        
        # Convert to DataFrame for better display
        outdated_df = pd.DataFrame(outdated_entries)
        
        # Add a last updated days ago column if it exists
        if 'last_updated' in outdated_df.columns:
            outdated_df['last_updated'] = pd.to_datetime(outdated_df['last_updated'])
            outdated_df['days_since_update'] = (pd.Timestamp.now() - outdated_df['last_updated']).dt.days
            
            # Sort by most outdated
            outdated_df = outdated_df.sort_values('days_since_update', ascending=False)
        
        # Select the most important columns to display
        display_cols = ['question', 'days_since_update', 'last_updated'] if 'question' in outdated_df.columns else outdated_df.columns[:3]
        st.dataframe(outdated_df[display_cols], use_container_width=True)
        
        # Individual entries
        for i, (_, entry) in enumerate(outdated_df.iterrows()):
            with st.expander(f"Entry {i+1}: Last updated {entry.get('days_since_update', 'Unknown')} days ago"):
                # Display entry details
                st.markdown(f"### Entry Details")
                
                # Format fields nicely
                for col, value in entry.items():
                    if col in ['question', 'answer']:
                        st.markdown(f"**{col.capitalize()}**:")
                        st.markdown(f"{value}")
                    elif col == 'last_updated':
                        st.markdown(f"**Last Updated**: {value}")
                
                # Action buttons
                st.markdown("---")
                action_col1, action_col2, action_col3 = st.columns(3)
                
                with action_col1:
                    if st.button(f"Mark as Updated {i+1}", key=f"update_{i}"):
                        st.success("Entry marked as updated (action not implemented)")
                
                with action_col2:
                    if st.button(f"Flag for Review {i+1}", key=f"flag_out_{i}"):
                        st.warning("Entry flagged for review (action not implemented)")
                
                with action_col3:
                    if st.button(f"Archive Entry {i+1}", key=f"archive_{i}"):
                        st.error("Entry archived (action not implemented)")
    else:
        st.warning("No outdated entries found matching the criteria.")

elif page == "Review Panel":
    st.markdown("<h1 class='main-header'>Review Panel</h1>", unsafe_allow_html=True)
    
    # In a real implementation, there would be a queue of items for review
    # For now, we'll create a sample interface
    
    st.markdown("<h2 class='sub-header'>Items Flagged for Review</h2>", unsafe_allow_html=True)
    
    # Mock data for demonstration
    review_items = [
        {"id": "item1", "type": "Cluster", "reason": "Low similarity score", "priority": "High"},
        {"id": "item2", "type": "Entry", "reason": "Outdated content", "priority": "Medium"},
        {"id": "item3", "type": "Pair", "reason": "Potential duplicate", "priority": "High"}
    ]
    
    # Convert to DataFrame for display
    review_df = pd.DataFrame(review_items)
    st.dataframe(review_df, use_container_width=True)
    
    # Side-by-side comparison view
    st.markdown("<h2 class='sub-header'>Side-by-Side Comparison</h2>", unsafe_allow_html=True)
    
    # Select items to compare
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Entry 1")
        st.markdown("""
        **Question**: What are the security measures in place for data access?
        
        **Answer**: Our system implements role-based access control (RBAC) to manage data access permissions. All data requests are logged and audited. We use encryption for data at rest and in transit, and employ multi-factor authentication for sensitive operations.
        
        **Last Updated**: 2022-05-15
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Entry 2")
        st.markdown("""
        **Question**: How does your system control access to data?
        
        **Answer**: We use role-based access control to manage permissions. All access is logged. Data is encrypted both at rest and in transit, and we require multi-factor authentication for sensitive operations.
        
        **Last Updated**: 2023-01-10
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Similarity metric
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Similarity Analysis")
    st.markdown("**Similarity Score**: 0.89")
    
    # Text differences could be highlighted here
    st.markdown("**Key Differences**:")
    st.markdown("- Different question phrasing but similar intent")
    st.markdown("- Second answer is more concise")
    st.markdown("- Content is essentially the same")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Decision buttons
    st.markdown("### Decision")
    decision_col1, decision_col2, decision_col3, decision_col4 = st.columns(4)
    
    with decision_col1:
        if st.button("Merge Entries"):
            st.success("Entries would be merged (action not implemented)")
    
    with decision_col2:
        if st.button("Keep Both"):
            st.info("Both entries will be kept (action not implemented)")
    
    with decision_col3:
        if st.button("Edit & Merge"):
            st.warning("Opening edit interface (action not implemented)")
    
    with decision_col4:
        if st.button("Defer Decision"):
            st.error("Decision deferred (action not implemented)")

elif page == "Pipeline Control":
    st.markdown("<h1 class='main-header'>Pipeline Control</h1>", unsafe_allow_html=True)
    
    # Get pipeline steps
    pipeline_steps = fetch_api_data("/pipeline/steps")
    
    if pipeline_steps:
        # Pipeline configuration
        st.markdown("<h2 class='sub-header'>Pipeline Configuration</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Product selection
            products = fetch_api_data("/products")
            if products:
                product_options = ["All Products"] + [p["product_name"] for p in products]
                pipeline_product = st.selectbox("Select Product", product_options)
                selected_product = None if pipeline_product == "All Products" else pipeline_product
            else:
                selected_product = None
                st.warning("Unable to load products")
        
        with col2:
            # Dry run option
            dry_run = st.checkbox("Dry Run (Preview Only)", value=False)
        
        # Step selection
        st.markdown("<h3>Pipeline Steps</h3>", unsafe_allow_html=True)
        
        # Display steps with checkboxes
        selected_steps = []
        for i, step in enumerate(pipeline_steps):
            if st.checkbox(f"{i+1}. {step['name']} ({step['file']})", value=True):
                selected_steps.append(i)
        
        if selected_steps:
            start_step = min(selected_steps)
            end_step = max(selected_steps) + 1  # +1 because the API expects exclusive end
        else:
            start_step = 0
            end_step = 0
        
        # Execute button
        if st.button("Run Pipeline"):
            options = {
                "product": selected_product,
                "start_step": start_step,
                "end_step": end_step,
                "dry_run": dry_run
            }
            
            result = run_pipeline(options)
            
            if result:
                st.success(f"Pipeline started: {result['message']}")
                st.json(result['options'])
                
                # In a real implementation, we might poll for updates
                st.info("Check logs for progress updates")
            else:
                st.error("Failed to start pipeline")
        
        # Schedule option (mock)
        st.markdown("<h2 class='sub-header'>Schedule Pipeline</h2>", unsafe_allow_html=True)
        
        schedule_col1, schedule_col2, schedule_col3 = st.columns(3)
        
        with schedule_col1:
            schedule_type = st.selectbox("Schedule Type", ["Daily", "Weekly", "Monthly"])
        
        with schedule_col2:
            if schedule_type == "Daily":
                schedule_time = st.time_input("Time", value=datetime.strptime("01:00", "%H:%M").time())
            elif schedule_type == "Weekly":
                schedule_day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                schedule_time = st.time_input("Time", value=datetime.strptime("01:00", "%H:%M").time())
            else:  # Monthly
                schedule_day = st.number_input("Day of Month", min_value=1, max_value=28, value=1)
                schedule_time = st.time_input("Time", value=datetime.strptime("01:00", "%H:%M").time())
        
        with schedule_col3:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            if st.button("Set Schedule"):
                st.success("Schedule set (action not implemented)")
                
        # Show logs
        st.markdown("<h2 class='sub-header'>Pipeline Logs</h2>", unsafe_allow_html=True)
        
        log_tabs = st.tabs(["Pipeline Trigger", "Embedding", "Clustering", "Cluster Grouping"])
        
        with log_tabs[0]:
            st.text("Loading pipeline_trigger.log...")
            # In a real implementation, we would load the actual log file
            st.code("""
2023-08-15 10:15:23 - pipeline_trigger - INFO - Starting pipeline at step 0 and ending at step 6
2023-08-15 10:15:23 - pipeline_trigger - INFO - Running Timed.py: /usr/bin/python3 /app/periodic_script/timed.py
2023-08-15 10:15:24 - pipeline_trigger - INFO - Successfully completed Timed.py
2023-08-15 10:15:24 - pipeline_trigger - INFO - Running Embedding: /usr/bin/python3 /app/periodic_script/embedding.py
2023-08-15 10:25:45 - pipeline_trigger - INFO - Successfully completed Embedding
            """)
        
        with log_tabs[1]:
            st.text("Loading embedding_pipeline.log...")
            st.code("""
2023-08-15 10:15:24 - embedding - INFO - Starting embedding pipeline
2023-08-15 10:15:30 - embedding - INFO - Loading model: all-MiniLM-L6-v2
2023-08-15 10:16:10 - embedding - INFO - Processing 53026 questions
2023-08-15 10:25:40 - embedding - INFO - Embedding pipeline completed
            """)
        
        with log_tabs[2]:
            st.text("Loading clustering_pipeline.log...")
            st.code("""
2023-08-15 10:25:46 - clustering - INFO - Starting clustering pipeline
2023-08-15 10:25:50 - clustering - INFO - Using HDBSCAN clustering algorithm
2023-08-15 10:26:30 - clustering - INFO - Found 3918 clusters
2023-08-15 10:26:45 - clustering - INFO - Clustering pipeline completed
            """)
        
        with log_tabs[3]:
            st.text("Loading cluster_grouping.log...")
            st.code("""
2023-08-15 10:26:50 - cluster_grouping - INFO - Starting cluster grouping pipeline
2023-08-15 10:26:55 - cluster_grouping - INFO - Grouping similar clusters
2023-08-15 10:27:30 - cluster_grouping - INFO - Cluster grouping pipeline completed
            """)
    else:
        st.warning("Unable to load pipeline steps. Please ensure the backend API is running.")

# Footer
st.markdown("---")
st.markdown("*QnA Content Management Dashboard v1.0 | Last updated: August 2023*") 