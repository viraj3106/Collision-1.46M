import os
import sys
import time
import subprocess
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from playground.api_client import CollisionAPIClient

def run_integration_test():
    print("Starting API backend server for integration tests...")
    # Start FastAPI backend server in background
    cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8085"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for startup
    time.sleep(3.0)
    
    try:
        # 1. Directly call API using requests
        payload = {
            "model": "collision-10m",
            "prompt": "What is artificial intelligence?",
            "max_tokens": 100,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9
        }
        
        print("Calling REST API backend directly...")
        api_res = requests.post("http://127.0.0.1:8085/v1/generate", json=payload, timeout=10.0)
        assert api_res.status_code == 200, f"API failed with {api_res.status_code}"
        api_data = api_res.json()
        
        # 2. Call via APIClient (Playground integration wrapper)
        print("Calling REST API backend via playground client wrapper...")
        client = CollisionAPIClient(base_url="http://127.0.0.1:8085")
        
        # Get health
        health = client.get_health()
        assert health["status"] == "ok", "Playground health check integration failed"
        
        # Get models
        models = client.get_models()
        assert "data" in models, "Playground models endpoint integration failed"
        
        # Generate
        client_res = client.generate(
            prompt="What is artificial intelligence?",
            model="collision-10m",
            max_tokens=100,
            temp=0.7,
            top_k=50,
            top_p=0.9
        )
        assert client_res["success"], f"Playground client generation failed: {client_res.get('error')}"
        client_data = client_res["data"]
        
        print("\nChecking alignment...")
        print(f"Direct text: '{api_data['text']}'")
        print(f"Client text: '{client_data['text']}'")
        
        # Verify that output alignment is verified (note: due to sampling, text may differ unless seeds are identical, but structure must match exactly)
        assert "id" in client_data
        assert "text" in client_data
        assert client_data["model"] == "collision-10m"
        assert "usage" in client_data
        assert "performance" in client_data
        
        print("Integration test passed successfully!")
        
    finally:
        print("Stopping backend server...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    run_integration_test()
