#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("content_updates.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("content_update_pipeline")

class ContentUpdatePipeline:
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self.cleaned_dir = self.root_dir / "cleaned_dataset"
        self.processed_dir = self.root_dir / "processed_clusters"
        
        # Ensure directories exist
        self.cleaned_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
    
    def load_data(self):
        """Load the current data files"""
        try:
            # Load main data file from cleaned_dataset
            data_file = self.cleaned_dir / "all_complete_dataset.csv"
            if not data_file.exists():
                logger.error(f"Main dataset not found at: {data_file}")
                return False
            self.df = pd.read_csv(data_file)
            
            # Load clusters data from processed_clusters
            clusters_file = self.processed_dir / "all_clusters.json"
            if not clusters_file.exists():
                logger.info("Clusters file not found, initializing with empty data")
                self.clusters = {"clusters": []}
            else:
                with open(clusters_file, 'r') as f:
                    self.clusters = json.load(f)
            
            logger.info("Successfully loaded all data files")
            return True
        except Exception as e:
            logger.error(f"Error loading data files: {e}")
            return False
    
    def update_content(self, cluster_id, updates):
        """Update content for a specific cluster"""
        try:
            # Update the dataframe
            mask = self.df['cluster_id'] == cluster_id
            
            if 'question' in updates:
                self.df.loc[mask, 'question'] = updates['question']
            if 'answer' in updates:
                self.df.loc[mask, 'answer'] = updates['answer']
            if 'details' in updates:
                self.df.loc[mask, 'details'] = updates['details']
            
            # Update last modified timestamp
            self.df.loc[mask, 'last_modified'] = datetime.now().isoformat()
            
            # Update clusters data
            for cluster in self.clusters.get("clusters", []):
                if cluster['cluster_id'] == cluster_id:
                    cluster.update(updates)
                    if 'health_status' not in updates:
                        cluster['health_status'] = 'Healthy'  # Mark as healthy after update
                    break
            
            logger.info(f"Successfully updated content for cluster {cluster_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating content for cluster {cluster_id}: {e}")
            return False
    
    def mark_reviewed(self, cluster_id):
        """Mark a cluster as reviewed"""
        try:
            # Update clusters data
            for cluster in self.clusters.get("clusters", []):
                if cluster['cluster_id'] == cluster_id:
                    cluster['last_reviewed'] = datetime.now().isoformat()
                    cluster['health_status'] = 'Healthy'
                    break
            
            logger.info(f"Successfully marked cluster {cluster_id} as reviewed")
            return True
        except Exception as e:
            logger.error(f"Error marking cluster {cluster_id} as reviewed: {e}")
            return False
    
    def save_changes(self):
        """Save all changes to files"""
        try:
            # Save main data file
            self.df.to_csv(self.cleaned_dir / "all_complete_dataset.csv", index=False)
            
            # Save clusters data
            with open(self.processed_dir / "all_clusters.json", 'w') as f:
                json.dump(self.clusters, f, indent=2)
            
            # Update summary
            self._update_summary()
            
            logger.info("Successfully saved all changes")
            return True
        except Exception as e:
            logger.error(f"Error saving changes: {e}")
            return False
    
    def _update_summary(self):
        """Update the summary statistics"""
        try:
            clusters = self.clusters.get("clusters", [])
            summary = {
                "total_questions": len(self.df),
                "total_clusters": len(clusters),
                "similar_clusters": sum(1 for c in clusters if c.get("is_canonical", False)),
                "outdated_content": sum(1 for c in clusters if c.get("health_status") == "Critical"),
                "health_score": self._calculate_health_score(clusters),
                "cluster_health": self._get_health_distribution(clusters),
                "avg_cluster_size": len(self.df) / len(clusters) if len(clusters) > 0 else 0
            }
            
            with open(self.processed_dir / "all_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("Successfully updated summary statistics")
        except Exception as e:
            logger.error(f"Error updating summary statistics: {e}")
    
    def _calculate_health_score(self, clusters):
        """Calculate overall health score"""
        total = len(clusters)
        if total == 0:
            return 100
        
        healthy = sum(1 for c in clusters if c.get("health_status") == "Healthy")
        return (healthy / total) * 100
    
    def _get_health_distribution(self, clusters):
        """Get distribution of health statuses"""
        distribution = {
            "Critical": 0,
            "Needs Review": 0,
            "Healthy": 0
        }
        
        for cluster in clusters:
            status = cluster.get("health_status", "Needs Review")
            distribution[status] = distribution.get(status, 0) + 1
        
        return distribution

def main():
    pipeline = ContentUpdatePipeline()
    if not pipeline.load_data():
        return 1
    
    # Example usage:
    # pipeline.update_content(1, {
    #     "question": "Updated question",
    #     "answer": "Updated answer",
    #     "details": "Updated details"
    # })
    # pipeline.mark_reviewed(1)
    # pipeline.save_changes()
    
    return 0

if __name__ == "__main__":
    exit(main()) 