import os
import sys
import torch
import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app
from api.dependencies import get_inference_engine

client = TestClient(app)

def run_regression_test():
    print("Initializing regression tests...")
    engine = get_inference_engine()
    
    prompts = [
        "What is artificial intelligence?",
        "Computer science is",
        "The future of technology",
        "An algorithm is",
        "Space exploration",
        "Why does the Earth orbit the Sun?",
        "Machine learning is",
        "Photosynthesis is"
    ]
    
    for idx, prompt in enumerate(prompts):
        print(f"Comparing prompt {idx+1}: '{prompt}'")
        
        # 1. Direct Inference
        torch.manual_seed(1337)
        np.random.seed(1337)
        direct_res = engine.generate(prompt, max_tokens=15, temp=0.8, top_k=50, top_p=0.9)
        
        # 2. API Inference
        torch.manual_seed(1337)
        np.random.seed(1337)
        api_res = client.post("/v1/generate", json={
            "model": "collision-10m",
            "prompt": prompt,
            "max_tokens": 15,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9
        })
        
        assert api_res.status_code == 200, f"API failed with {api_res.status_code}"
        api_data = api_res.json()
        
        # Verify equivalency
        assert direct_res["text"] == api_data["text"], f"Mismatch for '{prompt}':\nDirect: {direct_res['text']}\nAPI: {api_data['text']}"
        assert direct_res["completion_tokens"] == api_data["usage"]["completion_tokens"], f"Completion tokens mismatch"
        assert direct_res["prompt_tokens"] == api_data["usage"]["prompt_tokens"], f"Prompt tokens mismatch"
        
        print(" -> Output and tokens matched exactly.")

    print("\nGeneration regression test passed successfully!")

if __name__ == "__main__":
    run_regression_test()
