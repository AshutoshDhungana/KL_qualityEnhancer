# QnA Content Management Dashboard

This project provides a comprehensive dashboard for managing QnA content, including clustering analysis, similarity detection, and content optimization.

## Architecture

The system consists of two main components:

1. **FastAPI Backend**: Provides efficient API endpoints for accessing data and running the pipeline
2. **Streamlit Frontend**: User-friendly dashboard for visualizing and interacting with the data

## Features

- **Cluster Visualization**: View and explore clusters of similar QnA content
- **Similarity Analysis**: Compare entries side-by-side with similarity scores
- **Outdated Content Detection**: Identify and manage outdated entries
- **Interactive Review Panel**: Approve, reject, or merge similar content
- **Pipeline Control**: Run the analysis pipeline on-demand

## Installation

### Prerequisites

- Python 3.8 or higher
- Required Python packages (listed in requirements.txt)

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

### Option 1: All-in-One Launcher

The simplest way to run the application is using the provided launcher script:

```bash
python run.py
```

This will start both the FastAPI backend and Streamlit frontend.

- Backend will be available at: http://localhost:8000
- API documentation: http://localhost:8000/docs
- Frontend will be available at: http://localhost:8501

You can also start components individually:

```bash
# Start only the backend
python run.py --backend-only

# Start only the frontend
python run.py --frontend-only
```

### Option 2: Manual Start

If you prefer to start the components manually:

1. Start the FastAPI backend:

```bash
cd backend
python run_api.py
```

2. Start the Streamlit frontend:

```bash
streamlit run app.py
```

## Components

### Backend API Endpoints

The FastAPI backend provides the following key endpoints:

- `/summary` - Get overall summary statistics
- `/products` - List available products
- `/clusters` - Get clusters with optional filtering
- `/clusters/{cluster_id}` - Get detailed information about a specific cluster
- `/cluster/{cluster_id}/entries` - Get all QnA entries in a specific cluster
- `/outdated` - Get potentially outdated entries
- `/similar-pairs` - Get pairs of entries with high similarity scores
- `/pipeline/run` - Run the pipeline with specified options
- `/visualizations/*` - Various visualization endpoints

Full API documentation is available at http://localhost:8000/docs when the backend is running.

### Streamlit Dashboard Pages

The Streamlit frontend includes the following main pages:

1. **Dashboard** - Overview with key metrics and health statistics
2. **Cluster Explorer** - Explore and analyze content clusters
3. **Similarity Analysis** - Compare similar entries side-by-side
4. **Outdated Content** - Identify and manage outdated content
5. **Review Panel** - Interactive interface for review decisions
6. **Pipeline Control** - Run and schedule pipeline processes

## Pipeline Integration

The dashboard integrates with the existing ML pipeline that includes:

1. Embedding generation
2. Clustering
3. Cluster grouping
4. Cleaned dataset creation
5. Cluster cache generation

You can trigger the pipeline from the Pipeline Control page in the dashboard.

## Customization

### Changing Port Numbers

If you need to change the default ports:

- For the backend, modify the port in `backend/run_api.py`
- For the frontend, specify a different port when running Streamlit:
  ```bash
  streamlit run app.py --server.port <port-number>
  ```

### Changing API URLs

If the backend is running on a different machine or port, update the API URL in `app.py`:

```python
# Configuration
API_URL = "http://your-backend-host:port"
```

## Performance Considerations

- For large datasets, the API uses pagination and chunked loading for better performance
- The Streamlit frontend uses caching to minimize redundant API calls
- Consider running the backend and frontend on separate machines for heavy workloads

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
