# QnA Content Management Dashboard

A Streamlit-based dashboard for managing and analyzing QnA content using machine learning techniques. The dashboard provides clustering analysis, similarity detection, and content review features.

## Features

1. **Clustering Analysis**

   - Visualize clusters of similar QnA pairs
   - Interactive cluster exploration
   - Detailed view of cluster contents

2. **Similarity Analysis**

   - Heatmap visualization of pairwise similarities
   - Configurable similarity threshold
   - Side-by-side comparison of similar entries

3. **Outdated Content Detection**

   - List potentially outdated entries
   - Sorting and filtering options
   - Metadata-based analysis

4. **Interactive Review Panel**

   - Compare entries side by side
   - Approve/reject suggestions
   - Flag entries for updates

5. **Automated Analysis**
   - On-demand analysis pipeline
   - Scheduled periodic scans
   - Cached results for better performance

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Streamlit server:

```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Project Structure

```
.
├── README.md
├── requirements.txt
├── app.py                 # Main Streamlit application
└── backend/
    └── pipeline.py       # Backend processing pipeline
```

## Usage

1. **Running Analysis**

   - Click "Run Full Analysis" in the sidebar to process all QnA content
   - View results in the different tabs

2. **Scheduling Periodic Scans**

   - Set the desired interval in hours
   - Click "Schedule Periodic Scan" to enable automatic analysis

3. **Reviewing Content**
   - Use the tabs to navigate different views
   - Compare entries in the Review Panel
   - Take actions on similar or outdated content

## Customization

The sample data in `load_sample_data()` should be replaced with your actual data source. Modify the function in `app.py` to connect to your database or file system.

## Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- plotly
- scipy
- nltk
- python-dateutil
- apscheduler

## Notes

- The embedding computation in the backend pipeline is currently using random vectors for demonstration. Replace it with your preferred embedding model (e.g., sentence-transformers).
- Results are cached to improve performance. Clear the cache if you need to force a fresh analysis.
- The scheduler runs in the background and persists only while the application is running.
