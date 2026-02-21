"""
Simple test script to check if the API dependencies and code can initialize properly.
"""
import os
import sys
import traceback

def test_api_import():
    try:
        print("Step 1: Adding parent directory to path...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        
        print("\nStep 2: Attempting to import FastAPI and Uvicorn...")
        import fastapi
        import uvicorn
        print(f"FastAPI version: {fastapi.__version__}")
        print(f"Uvicorn version: {uvicorn.__version__}")
        
        print("\nStep 3: Attempting to import the app from backend/api.py...")
        try:
            sys.path.append(os.path.join(current_dir, 'backend'))
            from backend.api import app
            print("Successfully imported app from backend/api.py")
            
            # Print some basic info about the app
            print(f"App routes: {len(app.routes)}")
            for route in app.routes:
                print(f" - {route.path}")
                
            return True
        except Exception as e:
            print(f"Error importing app from backend/api.py: {str(e)}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"General error: {str(e)}")
        traceback.print_exc()
        return False

def test_merge_utils():
    try:
        print("\nStep 4: Testing merge_utils import...")
        import merge_utils
        print("Successfully imported merge_utils")
        
        print("\nStep 5: Testing QuestionMerger initialization...")
        merger = merge_utils.QuestionMerger()
        print("Successfully initialized QuestionMerger")
        
        return True
    except Exception as e:
        print(f"Error in merge_utils test: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("API Test Script")
    print("=" * 50)
    
    api_test_success = test_api_import()
    merge_utils_success = test_merge_utils()
    
    print("\nTest Results:")
    print(f"API Import Test: {'SUCCESS' if api_test_success else 'FAILED'}")
    print(f"Merge Utils Test: {'SUCCESS' if merge_utils_success else 'FAILED'}")
    
    if not api_test_success or not merge_utils_success:
        print("\nSuggested fixes:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check for import errors in the code files")
        print("3. Verify file paths and directory structure")
        sys.exit(1)
    else:
        print("\nAll tests passed successfully!")
        sys.exit(0) 