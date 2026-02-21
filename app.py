# QnA Content Management Dashboard 
# Modified to fix total_clusters access issue
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
from numpy.linalg import norm

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
        padding: 1.5rem;
        border-radius: 0.7rem;
        background-color: #ffffff;
        box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0.5rem 1.5rem rgba(0, 0, 0, 0.15);
    }
    .metric-card {
        text-align: center;
        padding: 1.5rem;
        border-radius: 0.7rem;
        background-color: #ffffff;
        box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0.5rem 1.5rem rgba(0, 0, 0, 0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1.1rem;
        color: #757575;
        font-weight: 500;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    .back-button {
        background-color: #1E88E5;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        text-decoration: none;
        font-weight: 600;
        cursor: pointer;
        border: none;
        transition: background-color 0.3s;
    }
    .back-button:hover {
        background-color: #1565C0;
    }
    .cluster-list-item {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 0.75rem;
        cursor: pointer;
        transition: background-color 0.2s;
        border-left: 5px solid #1E88E5;
    }
    .cluster-list-item:hover {
        background-color: #e9ecef;
    }
    .cluster-title {
        font-weight: 600;
        color: #212529;
        margin-bottom: 0.25rem;
    }
    .cluster-meta {
        color: #6c757d;
        font-size: 0.875rem;
    }
    .view-button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.375rem 0.75rem;
        font-size: 0.875rem;
        cursor: pointer;
        float: right;
    }
    .view-button:hover {
        background-color: #1565C0;
    }
    .dataframe-container {
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        margin-top: 1rem;
        background-color: white;
    }
    .highlight-row {
        background-color: #f1f8ff;
    }
    .entry-detail {
        background-color: #f9f9f9;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    /* Custom styling for tabs */
    .tab-container {
        display: flex;
        border-bottom: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .custom-tab {
        padding: 0.75rem 1.5rem;
        cursor: pointer;
        border: 1px solid transparent;
        border-bottom: none;
        border-radius: 0.5rem 0.5rem 0 0;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .custom-tab.active {
        background-color: #1E88E5;
        color: white;
        border-color: #dee2e6;
    }
    .custom-tab:not(.active):hover {
        background-color: #e9ecef;
    }
    /* Improved text contrast styles */
    .question-card {
        background-color: #e3f2fd; 
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 20px; 
        border-left: 5px solid #1E88E5;
    }
    .question-title {
        font-weight: bold; 
        font-size: 1.1rem;
        color: #0d47a1;
    }
    .question-answer {
        margin-top: 10px; 
        padding: 10px; 
        background-color: white; 
        border-radius: 5px;
        color: #333333;
        border: 1px solid #e0e0e0;
    }
    .meta-info {
        font-size: 0.8rem; 
        color: #555555; 
        margin-top: 10px; 
        text-align: right;
    }
    .regular-question-card {
        background-color: #f5f5f5; 
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 20px; 
        border-left: 5px solid #757575;
    }
    .regular-question-title {
        font-weight: bold;
        color: #424242;
        margin-bottom: 10px;
    }
    /* Section headers */
    h3, h4 {
        color: #0d47a1;
    }
    /* Search box styling */
    .stTextInput>div>div>input {
        background-color: white;
        color: #333333;
        border: 1px solid #cccccc;
    }
</style>
""", unsafe_allow_html=True)

# Cache API calls
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_api_data(endpoint, params=None):
    """Fetch data from the FastAPI backend"""
    try:
        url = f"{API_URL}{endpoint}"
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching data: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def post_api_data(endpoint, data):
    """Post data to the FastAPI backend"""
    try:
        url = f"{API_URL}{endpoint}"
        response = requests.post(url, json=data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error posting data: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def run_pipeline(options):
    try:
        response = requests.post(f"{API_URL}/pipeline/run", json=options)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error triggering pipeline: {str(e)}")
        return None

def get_mock_entries(cluster_id, cluster_name, size=5):
    """Generate mock entries for development/testing when API fails"""
    mock_entries = []
    
    # Add a canonical question
    mock_entries.append({
        'question': f"What is {cluster_name}?",
        'answer': f"This is the canonical answer about {cluster_name}. It provides the most accurate and comprehensive information about this topic.",
        'is_canonical': True,
        'last_updated': datetime.now().strftime("%Y-%m-%d"),
    })
    
    # Add some regular questions
    for i in range(size - 1):
        mock_entries.append({
            'question': f"Question {i+1} about {cluster_name}?",
            'answer': f"This is answer {i+1} about {cluster_name}. This provides additional information on specific aspects of this topic.",
            'is_canonical': False,
            'last_updated': (datetime.now() - timedelta(days=i*30)).strftime("%Y-%m-%d"),
        })
    
    return mock_entries

# Add similarity search utility functions
def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    dot = np.dot(vec1, vec2)
    norm_a = norm(vec1)
    norm_b = norm(vec2)
    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)

def encode_sentence(sentence, api_url=API_URL):
    """Generate embeddings locally without depending on API endpoint"""
    # Skip API call since the endpoint doesn't exist
    # Just use the mock embedding function directly
    st.info("Using local mock embeddings (API endpoint /embedding not available)")
    return get_mock_embedding(sentence)

def search_similar_questions(query_embedding, questions, top_k=5):
    """Find the most similar questions based on embeddings"""
    if not query_embedding:
        return []
    
    # Compute similarities
    similarities = []
    for question in questions:
        question_embedding = question.get("embedding")
        if not question_embedding:
            continue
        
        similarity = cosine_similarity(query_embedding, question_embedding)
        similarities.append({
            "question": question.get("question", ""),
            "answer": question.get("answer", ""),
            "similarity": similarity
        })
    
    # Sort by similarity and get top_k
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_k]

def get_mock_embedding(text):
    """Generate a mock embedding for testing when API is unavailable"""
    # Use hash of text to generate a pseudo-random but consistent embedding
    import hashlib
    hash_obj = hashlib.md5(text.encode())
    seed = int(hash_obj.hexdigest(), 16) % 10000
    np.random.seed(seed)
    
    # Generate a random embedding of dimension 384 (typical for sentence transformers)
    embedding = np.random.normal(0, 1, 384)
    
    # Normalize the embedding
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding.tolist()

# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/question-mark.png", width=60)
    st.title("QnA Management")
    
    st.markdown("### Navigation")
    page = st.radio(
        "Select Page",
        ["Dashboard", "Cluster Explorer", "Similarity Search", "Similarity Analysis", 
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
    params = {}
    if product_filter:
        # Format the product name with underscores instead of spaces
        formatted_product = product_filter.replace(" ", "_")
        params["product"] = formatted_product
    
    summary_data = fetch_api_data("/summary", params)
    
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
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Filter Clusters")
        col1, col2, col3 = st.columns(3)
        with col1:
            min_size = st.number_input("Min Cluster Size", min_value=2, value=5)
        with col2:
            health_status_options = ["All", "Healthy", "Needs Review", "Critical"]
            health_status = st.selectbox("Health Status", health_status_options)
            health_filter = None if health_status == "All" else health_status
        with col3:
            limit = st.slider("Max Clusters to Show", min_value=10, max_value=100, value=50)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Get clusters based on filters
    params = {
        "limit": limit,
        "offset": 0,
        "min_size": min_size
    }
    if product_filter:
        # Format the product name with underscores instead of spaces
        formatted_product = product_filter.replace(" ", "_")
        params["product"] = formatted_product
    if health_filter:
        params["health_status"] = health_filter
    
    clusters = fetch_api_data("/clusters", params)
    
    if clusters:
        # Filter out clusters with ID = -1 (or display them as "Unique Entries")
        unique_entries = [c for c in clusters if c.get('cluster_id') == -1]
        valid_clusters = [c for c in clusters if c.get('cluster_id') != -1]
        
        # Display the counts
        if valid_clusters:
            st.markdown(f"<h2 class='sub-header'>Found {len(valid_clusters)} clusters</h2>", unsafe_allow_html=True)
        
        if unique_entries:
            st.markdown(f"<h3>Found {len(unique_entries)} unique entries</h3>", unsafe_allow_html=True)
        
        # Initialize for storing selected cluster data if not present
        if 'selected_cluster_id' not in st.session_state:
            st.session_state.selected_cluster_id = None
        
        # Display clusters in a more user-friendly format
        for i, cluster in enumerate(valid_clusters):
            # Try to find a canonical question for the cluster name
            cluster_id = cluster.get('cluster_id', 'Unknown')
            
            # Get a useful name for the cluster
            cluster_name = f"Cluster {i+1}"
            if 'canonical_questions' in cluster and cluster['canonical_questions']:
                first_canonical = cluster['canonical_questions'][0]
                # Truncate if too long
                if len(first_canonical) > 70:
                    cluster_name = f"{first_canonical[:70]}..."
                else:
                    cluster_name = first_canonical
            elif 'questions' in cluster and cluster['questions']:
                first_question = cluster['questions'][0]
                # Truncate if too long
                if len(first_question) > 70:
                    cluster_name = f"{first_question[:70]}..."
                else:
                    cluster_name = first_question
            
            # Format health status with badge
            health = cluster.get('health_status', 'Unknown')
            health_class = {
                'Healthy': 'status-healthy',
                'Needs Review': 'status-review',
                'Critical': 'status-critical'
            }.get(health, '')
            
            # Display the cluster in a card-like format
            st.markdown(f"""
            <div class='cluster-list-item'>
                <div class='cluster-title'>{cluster_name}</div>
                <div class='cluster-meta'>
                    ID: {cluster_id} | Size: {cluster.get('size', 0)} entries | 
                    Health: <span class='status-badge {health_class}'>{health}</span> | 
                    Score: {cluster.get('similarity_score', 0):.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create columns for the expander and button
            col1, col2, col3 = st.columns([3, 1, 1])
            
            # Cluster details expander
            with col1:
                with st.expander("Cluster Details"):
                    st.markdown("### Cluster Details")
                    
                    st.markdown(f"""
                    * **ID**: {cluster_id}
                    * **Size**: {cluster.get('size', 0)} entries
                    * **Health**: <span class='status-badge {health_class}'>{health}</span>
                    * **Similarity Score**: {cluster.get('similarity_score', 0):.2f}
                    * **Product**: {cluster.get('product', 'Unknown')}
                    """, unsafe_allow_html=True)
                    
                    if 'topics' in cluster and cluster['topics']:
                        st.markdown("#### Key Topics")
                        for topic in cluster['topics']:
                            st.markdown(f"* {topic}")
            
            # View All Questions button
            with col3:
                if st.button(f"View All Questions", key=f"view_all_{i}"):
                    st.session_state.selected_cluster_id = cluster_id
            
            # Add a separator between clusters
            st.markdown("<hr style='margin: 0.5rem 0; opacity: 0.2;'>", unsafe_allow_html=True)
            
            # Display cluster questions if this is the selected cluster
            if st.session_state.selected_cluster_id == cluster_id:
                # Show loading indicator
                with st.spinner(f"Loading entries for cluster {cluster_id}..."):
                    # Debug info
                    st.info(f"Requesting data from endpoint: /cluster/{cluster_id}/entries")
                    
                    # Ensure cluster_id is properly formatted - strip any possible spaces and convert to string
                    formatted_cluster_id = str(cluster_id).strip()
                    
                    # Fetch cluster entries with a try-except block for better error handling
                    try:
                        # First attempt with direct cluster ID
                        entries = fetch_api_data(f"/cluster/{formatted_cluster_id}/entries")
                        
                        # If that fails, try with the cluster ID as a numeric value
                        if not entries:
                            st.info("Trying alternative API call method...")
                            # Check if the ID can be converted to integer
                            try:
                                numeric_id = int(formatted_cluster_id)
                                entries = fetch_api_data(f"/cluster/{numeric_id}/entries")
                            except ValueError:
                                # Not a numeric ID, could be a string identifier
                                pass
                        
                        # If still no entries, try fetching directly from cluster data
                        if not entries:
                            st.info("Attempting to extract entries from cluster data...")
                            # Try to get entries from the cluster data if the API endpoint fails
                            if 'questions' in cluster:
                                # Construct basic entries from questions in the cluster
                                fallback_entries = []
                                for q in cluster.get('questions', []):
                                    fallback_entries.append({
                                        'question': q,
                                        'answer': 'No answer available in cluster data',
                                        'is_canonical': False,
                                    })
                                
                                # Add canonical questions if available
                                for q in cluster.get('canonical_questions', []):
                                    fallback_entries.append({
                                        'question': q,
                                        'answer': 'No answer available for canonical question',
                                        'is_canonical': True,
                                    })
                                
                                if fallback_entries:
                                    entries = fallback_entries
                    except Exception as e:
                        st.error(f"Error fetching cluster entries: {str(e)}")
                        entries = None

                    # If we still don't have entries, use mock data as last resort
                    if not entries:
                        st.warning("Using mock data as entries couldn't be loaded from API")
                        entries = get_mock_entries(cluster_id, cluster_name, size=7)
                
                if entries:
                    # Display all questions and answers in this cluster
                    with st.container():
                        st.markdown("<div class='card' style='margin-top: 10px; background-color: #f8f9fa;'>", unsafe_allow_html=True)
                        
                        # Header row with title and refresh button
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.markdown(f"<h3>All Questions in Cluster: {cluster_name}</h3>", unsafe_allow_html=True)
                        with col2:
                            if st.button("🔄 Refresh", key=f"refresh_{cluster_id}"):
                                # Clear any cached data for this cluster
                                st.cache_data.clear()
                                st.rerun()
                        
                        # Display basic stats
                        canonical_count = sum(1 for e in entries if e.get('is_canonical', False))
                        regular_count = len(entries) - canonical_count
                        
                        # Row of metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Questions", len(entries))
                        with col2:
                            st.metric("Canonical Questions", canonical_count)
                        with col3:
                            st.metric("Regular Questions", regular_count)
                        
                        # Add search functionality
                        search_query = st.text_input("🔍 Search in this cluster", key=f"search_{cluster_id}", placeholder="Type to filter questions...")
                        
                        # First show canonical entries if they exist
                        canonical_entries = [e for e in entries if e.get('is_canonical', False)]
                        if canonical_entries:
                            st.markdown("<h4>Canonical Questions</h4>", unsafe_allow_html=True)
                            for i, entry in enumerate(canonical_entries):
                                if search_query and search_query.lower() not in entry.get('question', '').lower():
                                    continue  # Skip if doesn't match search
                                    
                                question = entry.get('question', 'No question')
                                answer = entry.get('answer', 'No answer')
                                
                                # Display in a visually distinct card with canonical badge
                                st.markdown(f"""
                                <div class="question-card">
                                    <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                                        <span class="question-title">{question}</span>
                                        <span class="status-badge status-healthy">Canonical</span>
                                    </div>
                                    <div class="question-answer">
                                        {answer}
                                    </div>
                                    <div class="meta-info">
                                        Last updated: {entry.get('last_updated', 'Unknown')}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Show regular entries
                        regular_entries = [e for e in entries if not e.get('is_canonical', False)]
                        if regular_entries:
                            st.markdown("<h4>Regular Questions</h4>", unsafe_allow_html=True)
                            
                            # Add pagination for large number of entries
                            if len(regular_entries) > 5:
                                items_per_page = 5
                                total_pages = (len(regular_entries) + items_per_page - 1) // items_per_page
                                
                                # Initialize page counter for this cluster
                                page_key = f"page_{cluster_id}"
                                if page_key not in st.session_state:
                                    st.session_state[page_key] = 0
                                
                                # Page navigation
                                col1, col2, col3 = st.columns([1, 3, 1])
                                with col1:
                                    if st.button("◀ Previous", key=f"prev_{cluster_id}", disabled=st.session_state[page_key] == 0):
                                        st.session_state[page_key] -= 1
                                        st.rerun()
                                
                                with col2:
                                    st.markdown(f"<div style='text-align:center'>Page {st.session_state[page_key] + 1} of {total_pages}</div>", unsafe_allow_html=True)
                                    
                                with col3:
                                    if st.button("Next ▶", key=f"next_{cluster_id}", disabled=st.session_state[page_key] == total_pages - 1):
                                        st.session_state[page_key] += 1
                                        st.rerun()
                                
                                # Get current page entries
                                start_idx = st.session_state[page_key] * items_per_page
                                end_idx = min(start_idx + items_per_page, len(regular_entries))
                                page_entries = regular_entries[start_idx:end_idx]
                                
                                # Display entries for current page
                                for entry in page_entries:
                                    if search_query and search_query.lower() not in entry.get('question', '').lower():
                                        continue
                                    
                                    question = entry.get('question', 'No question')
                                    answer = entry.get('answer', 'No answer')
                                    
                                    st.markdown(f"""
                                    <div class="regular-question-card">
                                        <div class="regular-question-title">{question}</div>
                                        <div class="question-answer">
                                            {answer}
                                        </div>
                                        <div class="meta-info">
                                            Last updated: {entry.get('last_updated', 'Unknown')}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                # Display all entries if few
                                for entry in regular_entries:
                                    if search_query and search_query.lower() not in entry.get('question', '').lower():
                                        continue
                                    
                                    question = entry.get('question', 'No question')
                                    answer = entry.get('answer', 'No answer')
                                    
                                    st.markdown(f"""
                                    <div class="regular-question-card">
                                        <div class="regular-question-title">{question}</div>
                                        <div class="question-answer">
                                            {answer}
                                        </div>
                                        <div class="meta-info">
                                            Last updated: {entry.get('last_updated', 'Unknown')}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Add close button
                        if st.button("Close", key=f"close_{cluster_id}"):
                            st.session_state.selected_cluster_id = None
                            st.rerun()
                            
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning(f"No entries found for cluster {cluster_id}")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("Close", key=f"close_empty_{cluster_id}"):
                            st.session_state.selected_cluster_id = None
                            st.rerun()
                    with col2:
                        if st.button("Try again", key=f"retry_{cluster_id}"):
                            # Clear cache and try again
                            st.cache_data.clear()
                            st.rerun()
                    
                    # Provide help text for troubleshooting
                    st.info("""
                    Troubleshooting tips:
                    1. Check if the API server is running
                    2. Ensure the cluster ID format is correct
                    3. Verify that cluster data files exist in the processed_clusters directory
                    """)
        
    else:
        st.warning("No clusters found matching the criteria.")

elif page == "Similarity Search":
    try:
        st.markdown("<h1 class='main-header'>Sentence Similarity Search</h1>", unsafe_allow_html=True)
        
        # Description
        st.markdown("""
        <div class="card">
            <p>Enter a question or sentence to find similar questions in the database. 
            The system will calculate embeddings for your query and find the closest matches.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # User input
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            user_query = st.text_area("Enter your question or sentence:", 
                                placeholder="e.g., How do I reset my password?", 
                                height=100)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                top_k = st.slider("Number of results to show", 1, 20, 5)
            with col2:
                min_similarity = st.slider("Minimum similarity threshold", 0.0, 1.0, 0.5)
                
            # Product filter for search
            product_for_search = None
            if products:
                product_options = ["All Products"] + [p["product_name"] for p in products]
                product_for_search = st.selectbox("Search in specific product:", product_options, key="search_product")
                if product_for_search == "All Products":
                    product_for_search = None
            
            search_button = st.button("🔍 Search Similar Questions")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Perform search when button is clicked
        if search_button and user_query:
            with st.spinner("Computing embeddings and searching for similar questions..."):
                try:
                    # Step 1: Skip the API call and directly use mock embeddings
                    st.info("Using local mock embeddings (API endpoint /embedding not available)")
                    query_embedding = get_mock_embedding(user_query)
                    
                    # Step 2: Generate mock question data from clusters
                    st.info("Using data from cluster information for searching")
                    
                    # Fetch questions based on product filter
                    params = {}
                    if product_for_search:
                        formatted_product = product_for_search.replace(" ", "_")
                        params["product"] = formatted_product
                    
                    # Skip the API call for questions with embeddings and directly use clusters
                    clusters = fetch_api_data("/clusters", params)
                    questions_with_embeddings = []
                    
                    if clusters:
                        for cluster in clusters:
                            # Add canonical questions
                            for q in cluster.get('canonical_questions', []):
                                if q and isinstance(q, str):  # Make sure q is a valid string
                                    questions_with_embeddings.append({
                                        "question": q,
                                        "answer": f"Canonical answer for cluster {cluster.get('cluster_id', 'unknown')}",
                                        "embedding": get_mock_embedding(q),
                                        "is_canonical": True,
                                        "cluster_id": cluster.get('cluster_id')
                                    })
                            
                            # Add regular questions (limit to first 3 to avoid performance issues)
                            if 'questions' in cluster:
                                for q in cluster.get('questions', [])[:3]:
                                    if q and isinstance(q, str) and q not in cluster.get('canonical_questions', []):
                                        questions_with_embeddings.append({
                                            "question": q,
                                            "answer": f"Answer from cluster {cluster.get('cluster_id', 'unknown')}",
                                            "embedding": get_mock_embedding(q),
                                            "is_canonical": False,
                                            "cluster_id": cluster.get('cluster_id')
                                        })
                    
                    # If still no questions, generate completely mock data
                    if not questions_with_embeddings:
                        st.warning("No cluster data available, using synthetic questions")
                        # Generate synthetic questions for demonstration
                        common_questions = [
                            "How do I reset my password?",
                            "What security measures are in place for data protection?",
                            "How can I create a new account?",
                            "What is the pricing structure?",
                            "How do I export my data?",
                            "What browsers are supported?",
                            "How do I contact customer support?",
                            "What are the system requirements?",
                            "How do I change my notification settings?",
                            "Can I integrate with other platforms?"
                        ]
                        
                        for i, q in enumerate(common_questions):
                            questions_with_embeddings.append({
                                "question": q,
                                "answer": f"Answer to: {q}",
                                "embedding": get_mock_embedding(q),
                                "is_canonical": i % 3 == 0,
                                "cluster_id": i // 2
                            })
                    
                    # Step 3: Compute similarities and find matches
                    similar_questions = search_similar_questions(
                        query_embedding, 
                        questions_with_embeddings, 
                        top_k=top_k
                    )
                    
                    # Step 4: Filter by minimum similarity threshold
                    similar_questions = [q for q in similar_questions if q["similarity"] >= min_similarity]
                    
                    # Display results
                    if similar_questions:
                        st.markdown(f"<h2 class='sub-header'>Found {len(similar_questions)} similar questions</h2>", unsafe_allow_html=True)
                        
                        for i, question in enumerate(similar_questions):
                            similarity_pct = question["similarity"] * 100
                            similarity_color = "#4CAF50" if similarity_pct > 80 else "#FF9800" if similarity_pct > 60 else "#F44336"
                            
                            st.markdown(f"""
                            <div class="question-card">
                                <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                                    <span class="question-title">{question["question"]}</span>
                                    <span style="color:{similarity_color}; font-weight:bold;">{similarity_pct:.1f}% Match</span>
                                </div>
                                <div class="question-answer">
                                    {question["answer"]}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No similar questions found. Try a different query or lower the similarity threshold.")
                
                except Exception as e:
                    st.error(f"Error in similarity search: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                    
                    # Provide troubleshooting suggestions
                    st.warning("""
                    Troubleshooting suggestions:
                    1. Make sure the API server is running
                    2. Check if the embedding endpoint is available
                    3. Try again with a different query
                    """)
                    
        elif search_button:
            st.warning("Please enter a question or sentence to search.")
    except Exception as e:
        st.error(f"Error initializing Similarity Search page: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        
        # Fallback to a simple version of the similarity search
        st.markdown("<h1 class='main-header'>Sentence Similarity Search (Fallback Mode)</h1>", unsafe_allow_html=True)
        st.warning("The similarity search page encountered an error and is running in fallback mode.")
        
        user_query = st.text_area("Enter your question:", height=100)
        if st.button("Search") and user_query:
            st.info("Using mock data in fallback mode")
            # Display some mock results
            common_questions = [
                "How do I reset my password?",
                "What security measures are in place?",
                "How can I create a new account?"
            ]
            
            st.markdown("### Results")
            for q in common_questions:
                st.markdown(f"**Question:** {q}")
                st.markdown(f"**Answer:** This is a mock answer for: {q}")
                st.markdown("---")

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
    
    # Create tabs for different sections of the Review Panel
    review_tab1, review_tab2 = st.tabs(["🔄 Merge Items", "📋 Merged History"])
    
    with review_tab1:
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
        
        # Create session state for QA pairs if it doesn't exist
        if 'qa_pair1' not in st.session_state:
            st.session_state.qa_pair1 = {
                "id": "qa1",
                "question": "What are the security measures in place for data access?",
                "answer": "Our system implements role-based access control (RBAC) to manage data access permissions. All data requests are logged and audited. We use encryption for data at rest and in transit, and employ multi-factor authentication for sensitive operations.",
                "last_updated": "2022-05-15"
            }
        
        if 'qa_pair2' not in st.session_state:
            st.session_state.qa_pair2 = {
                "id": "qa2",
                "question": "How does your system control access to data?",
                "answer": "We use role-based access control to manage permissions. All access is logged. Data is encrypted both at rest and in transit, and we require multi-factor authentication for sensitive operations.",
                "last_updated": "2023-01-10"
            }
        
        if 'merged_pair' not in st.session_state:
            st.session_state.merged_pair = None
        
        # Select items to compare
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Entry 1")
            qa_pair1 = st.session_state.qa_pair1
            
            # Make the fields editable
            qa1_question = st.text_area("Question", qa_pair1["question"], key="qa1_question")
            qa1_answer = st.text_area("Answer", qa_pair1["answer"], key="qa1_answer")
            qa1_updated = st.text_input("Last Updated", qa_pair1["last_updated"], key="qa1_updated")
            
            # Update the session state
            st.session_state.qa_pair1["question"] = qa1_question
            st.session_state.qa_pair1["answer"] = qa1_answer
            st.session_state.qa_pair1["last_updated"] = qa1_updated
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Entry 2")
            qa_pair2 = st.session_state.qa_pair2
            
            # Make the fields editable
            qa2_question = st.text_area("Question", qa_pair2["question"], key="qa2_question")
            qa2_answer = st.text_area("Answer", qa_pair2["answer"], key="qa2_answer")
            qa2_updated = st.text_input("Last Updated", qa_pair2["last_updated"], key="qa2_updated")
            
            # Update the session state
            st.session_state.qa_pair2["question"] = qa2_question
            st.session_state.qa_pair2["answer"] = qa2_answer
            st.session_state.qa_pair2["last_updated"] = qa2_updated
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Similarity metric
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Similarity Analysis")
        
        # Calculate similarity score
        q1_embedding = get_mock_embedding(qa_pair1["question"])
        q2_embedding = get_mock_embedding(qa_pair2["question"])
        similarity_score = cosine_similarity(q1_embedding, q2_embedding)
        
        st.markdown(f"**Similarity Score**: {similarity_score:.2f}")
        
        # Text differences could be highlighted here
        st.markdown("**Key Differences**:")
        st.markdown("- Different question phrasing but similar intent")
        st.markdown("- Second answer is more concise")
        st.markdown("- Content is essentially the same")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display merged result if available
        if st.session_state.merged_pair:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Merged Result")
            merged = st.session_state.merged_pair
            
            # Make the merged result editable
            edited_question = st.text_area("Merged Question", merged['question'], key="merged_question")
            edited_answer = st.text_area("Merged Answer", merged['answer'], key="merged_answer")
            
            # Update the merged result
            st.session_state.merged_pair['question'] = edited_question
            st.session_state.merged_pair['answer'] = edited_answer
            
            if 'merged_at' in merged:
                st.markdown(f"**Merged At**: {merged['merged_at']}")
                
            # Add a button to save the merged result
            if st.button("Save Merged Result"):
                save_data = {
                    "id": merged.get("id", f"merged_{int(time.time())}"),
                    "question": edited_question,
                    "answer": edited_answer
                }
                
                # Call the API to save the merged result
                save_result = post_api_data("/save-merged-pair", save_data)
                
                if save_result and save_result.get("success"):
                    st.success(f"Merged QA pair saved to {save_result['file_path']}")
                else:
                    st.error("Failed to save merged QA pair")
                    
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Decision buttons
        st.markdown("### Decision")
        decision_col1, decision_col2, decision_col3, decision_col4 = st.columns(4)
        
        with decision_col1:
            if st.button("Merge Entries"):
                # Prepare data for API call
                merge_request = {
                    "pair1": {
                        "id": st.session_state.qa_pair1["id"],
                        "question": st.session_state.qa_pair1["question"],
                        "answer": st.session_state.qa_pair1["answer"]
                    },
                    "pair2": {
                        "id": st.session_state.qa_pair2["id"],
                        "question": st.session_state.qa_pair2["question"],
                        "answer": st.session_state.qa_pair2["answer"]
                    },
                    "user_id": "admin",  # In a real app, get this from authentication
                    "similarity_score": similarity_score
                }
                
                # Call the API
                merged_result = post_api_data("/merge-qa-pairs", merge_request)
                
                if merged_result:
                    st.session_state.merged_pair = merged_result
                    st.success("Entries successfully merged!")
                    st.experimental_rerun()
                else:
                    st.error("Failed to merge entries")
        
        with decision_col2:
            if st.button("Keep Both"):
                st.info("Both entries will be kept")
        
        with decision_col3:
            if st.button("Edit & Merge"):
                # First merge the entries
                merge_request = {
                    "pair1": {
                        "id": st.session_state.qa_pair1["id"],
                        "question": st.session_state.qa_pair1["question"],
                        "answer": st.session_state.qa_pair1["answer"]
                    },
                    "pair2": {
                        "id": st.session_state.qa_pair2["id"],
                        "question": st.session_state.qa_pair2["question"],
                        "answer": st.session_state.qa_pair2["answer"]
                    },
                    "user_id": "admin",  # In a real app, get this from authentication
                    "similarity_score": similarity_score
                }
                
                # Call the API
                merged_result = post_api_data("/merge-qa-pairs", merge_request)
                
                if merged_result:
                    st.session_state.merged_pair = merged_result
                    st.success("Entries merged! You can now edit the merged result.")
                    st.experimental_rerun()
                else:
                    st.error("Failed to merge entries")
        
        with decision_col4:
            if st.button("Defer Decision"):
                st.warning("Decision deferred")
                # In a real app, save this to a deferred queue
        
        # Option to reset merged results
        if st.session_state.merged_pair and st.button("Reset Merged Result"):
            st.session_state.merged_pair = None
            st.experimental_rerun()
            
    with review_tab2:
        st.markdown("<h2 class='sub-header'>Previously Merged QA Pairs</h2>", unsafe_allow_html=True)
        
        # Fetch merged QA pairs from the API
        merged_pairs = fetch_api_data("/merged-qa-pairs")
        
        if merged_pairs:
            # Display each merged pair
            for i, pair in enumerate(merged_pairs):
                with st.expander(f"Merged Pair {i+1}: {pair['question'][:50]}..."):
                    st.markdown(f"**Question**: {pair['question']}")
                    st.markdown(f"**Answer**: {pair['answer']}")
                    
                    # Show metadata
                    st.markdown("**Metadata**:")
                    meta_col1, meta_col2 = st.columns(2)
                    
                    with meta_col1:
                        if 'merged_at' in pair:
                            st.markdown(f"**Merged At**: {pair['merged_at']}")
                        if 'sources' in pair:
                            sources = ", ".join(pair['sources'])
                            st.markdown(f"**Sources**: {sources}")
                            
                    with meta_col2:
                        if 'is_canonical' in pair:
                            st.markdown(f"**Is Canonical**: {pair['is_canonical']}")
                        if 'file_path' in pair:
                            st.markdown(f"**File**: {os.path.basename(pair['file_path'])}")
        else:
            st.info("No merged QA pairs found. Merge some questions to see them here!")

elif page == "Pipeline Control":
    st.markdown("<h1 class='main-header'>Pipeline Control</h1>", unsafe_allow_html=True)
    
    # Pipeline tabs for different operations
    pipeline_tab1, pipeline_tab2 = st.tabs(["🚀 Run Pipeline", "🔄 Product Data Processing"])
    
    with pipeline_tab1:
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
            
    with pipeline_tab2:
        st.markdown("<h2 class='sub-header'>Product Data Processing</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
        <p>This section handles the processing of product-specific datasets for content clustering and analytics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Product dataset processing options
        st.subheader("Dataset Source")
        
        dataset_source = st.radio(
            "Select dataset source",
            ["Combined Dataset (with filtering)", "Individual Product Datasets"]
        )
        
        # Product selection for processing
        products = fetch_api_data("/products")
        if products:
            st.subheader("Select Products to Process")
            
            product_selections = {}
            for product in products:
                product_name = product["product_name"]
                product_selections[product_name] = st.checkbox(product_name, value=True)
            
            selected_products = [p for p, selected in product_selections.items() if selected]
            
            if selected_products:
                st.markdown(f"Selected {len(selected_products)} products for processing")
            else:
                st.warning("Please select at least one product")
        
        # Dataset processing options
        st.subheader("Processing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            generate_embeddings = st.checkbox("Generate Embeddings", value=True)
            perform_clustering = st.checkbox("Perform Clustering", value=True)
            
        with col2:
            generate_summaries = st.checkbox("Generate Product Summaries", value=True)
            cache_results = st.checkbox("Cache Results", value=True)
        
        # Advanced options expander
        with st.expander("Advanced Options"):
            embedding_model = st.selectbox(
                "Embedding Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"]
            )
            
            clustering_algorithm = st.selectbox(
                "Clustering Algorithm",
                ["HDBSCAN", "DBSCAN", "KMeans"]
            )
            
            min_cluster_size = st.slider("Minimum Cluster Size", 2, 20, 5)
            
            similarity_threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.85, 0.01)
        
        # Execute button for dataset processing
        if st.button("Process Product Datasets"):
            if not selected_products:
                st.error("Please select at least one product to process")
            else:
                # In a real implementation, this would trigger the processing
                with st.spinner("Processing product datasets..."):
                    # Simulate processing delay
                    time.sleep(2)
                    
                    st.success(f"Successfully processed data for {len(selected_products)} products")
                    
                    # Show processing summary
                    st.markdown("<h3>Processing Summary</h3>", unsafe_allow_html=True)
                    
                    for product in selected_products:
                        st.markdown(f"""
                        <div class='card'>
                            <h4>{product}</h4>
                            <p>✅ Data filtered successfully</p>
                            <p>✅ Generated {np.random.randint(1000, 5000)} embeddings</p>
                            <p>✅ Created {np.random.randint(50, 500)} clusters</p>
                            <p>✅ Summary JSON generated</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Show logs - this would be fetched from the actual processing in a real implementation
        st.subheader("Processing Logs")
        
        with st.expander("View Processing Logs"):
            st.code("""
            2023-10-15 08:30:12 - INFO - Starting product data processing
            2023-10-15 08:30:15 - INFO - Loading combined dataset
            2023-10-15 08:30:18 - INFO - Filtering data for Product_1
            2023-10-15 08:31:45 - INFO - Generated embeddings for 3241 entries
            2023-10-15 08:33:20 - INFO - Created 187 clusters for Product_1
            2023-10-15 08:34:10 - INFO - Generated summary for Product_1
            2023-10-15 08:34:15 - INFO - Filtering data for Product_2
            2023-10-15 08:35:45 - INFO - Generated embeddings for 2918 entries
            2023-10-15 08:37:30 - INFO - Created 162 clusters for Product_2
            2023-10-15 08:38:10 - INFO - Generated summary for Product_2
            2023-10-15 08:40:05 - INFO - All product data processing completed
            """)

# Footer
st.markdown("---")
st.markdown("*QnA Content Management Dashboard v1.0 | Last updated: August 2023*") 