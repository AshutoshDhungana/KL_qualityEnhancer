"""
API Client Test Script to diagnose connection issues with more detailed error information.
"""
import sys
import time
import socket
import requests
from requests.exceptions import RequestException
import traceback

def check_port_open(host, port, timeout=2):
    """Check if a port is open on the given host"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def test_api_connection(url, max_retries=3, timeout=5):
    """Test connection to the API with retries and detailed error reporting"""
    print(f"Testing connection to {url}...")
    
    # Parse host and port from URL
    if url.startswith("http://"):
        url_parts = url[7:].split(":")
    elif url.startswith("https://"):
        url_parts = url[8:].split(":")
    else:
        url_parts = url.split(":")
    
    host = url_parts[0]
    port = int(url_parts[1].split("/")[0]) if len(url_parts) > 1 else 80
    
    # First check if the port is open
    print(f"Checking if port {port} is open on {host}...")
    if not check_port_open(host, port):
        print(f"Port {port} is not open on {host}. The server may not be running.")
        return False
    
    # Try to connect with retries
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}/{max_retries} to connect to {url}")
            response = requests.get(url, timeout=timeout)
            
            print(f"Connection successful! Status code: {response.status_code}")
            print(f"Response content: {response.text[:200]}...")
            return True
            
        except RequestException as e:
            print(f"Request failed: {str(e)}")
            if isinstance(e, requests.ConnectTimeout):
                print("Connection timed out. The server may be slow to respond.")
            elif isinstance(e, requests.ConnectionError):
                print("Connection error. The server may not be running or is unreachable.")
            
            # Wait before retrying
            if attempt < max_retries:
                wait_time = attempt * 2
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print("Maximum retry attempts reached.")
                return False
                
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            traceback.print_exc()
            return False
    
    return False

def try_api_endpoints(base_url):
    """Try different API endpoints"""
    endpoints = [
        "/",
        "/docs",
        "/summary",
        "/products"
    ]
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        print(f"\nTrying endpoint: {url}")
        test_api_connection(url, max_retries=1)

if __name__ == "__main__":
    print("API Client Test Script")
    print("=" * 50)
    
    # Test various hostnames and ports
    test_urls = [
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]
    
    for url in test_urls:
        print(f"\nTesting API at {url}")
        if test_api_connection(url):
            print(f"Successfully connected to {url}")
            try_api_endpoints(url)
            print("\nAPI connection test passed!")
            sys.exit(0)
        else:
            print(f"Failed to connect to {url}")
    
    print("\nAll connection attempts failed.")
    print("\nPossible causes:")
    print("1. The API server is not running")
    print("2. The API server is running on a different port")
    print("3. Firewall or network issues are blocking the connection")
    print("4. The API server is crashing on startup")
    print("\nSuggested solutions:")
    print("1. Start the API server: cd backend && python run_api.py")
    print("2. Check for error messages in the console when starting the server")
    print("3. Check if another process is using port 8000")
    print("4. Try running on a different port: add port=8001 to uvicorn.run() in run_api.py")
    sys.exit(1) 