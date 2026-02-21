#!/usr/bin/env python3
import os
import sys
import signal
import subprocess
import time
import argparse
from pathlib import Path

def run_backend():
    """Run the FastAPI backend"""
    print("Starting FastAPI backend...")
    return subprocess.Popen(
        [sys.executable, "backend/run_api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

def run_frontend():
    """Run the Streamlit frontend"""
    print("Starting Streamlit frontend...")
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.runOnSave", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

def main():
    parser = argparse.ArgumentParser(description="Run the QnA Content Management System")
    parser.add_argument("--backend-only", action="store_true", help="Run only the backend")
    parser.add_argument("--frontend-only", action="store_true", help="Run only the frontend")
    args = parser.parse_args()
    
    processes = []
    
    try:
        # Start the backend unless frontend-only is specified
        if not args.frontend_only:
            backend_process = run_backend()
            processes.append(("Backend", backend_process))
            # Wait a bit for the backend to start
            time.sleep(2)
        
        # Start the frontend unless backend-only is specified
        if not args.backend_only:
            frontend_process = run_frontend()
            processes.append(("Frontend", frontend_process))
        
        print("\n========= QnA Content Management System =========")
        if not args.frontend_only:
            print("Backend running at: http://localhost:8000")
            print("API docs available at: http://localhost:8000/docs")
        if not args.backend_only:
            print("Frontend running at: http://localhost:8501")
        print("==================================================\n")
        
        # Print output from processes in real-time
        try:
            while all(p[1].poll() is None for p in processes):
                for name, process in processes:
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            break
                        print(f"[{name}] {line.rstrip()}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        
    finally:
        # Clean up processes
        for name, process in processes:
            if process.poll() is None:
                print(f"Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing {name}...")
                    process.kill()

if __name__ == "__main__":
    main() 