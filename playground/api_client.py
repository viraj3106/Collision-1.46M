import requests
import os
import json

class CollisionAPIClient:
    def __init__(self, base_url=None, api_key=None, session_token=None):
        if base_url is None:
            base_url = os.environ.get("COLLISION_API_URL", "http://localhost:8000")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session_token = session_token

    def _get_api_headers(self):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_session_headers(self):
        headers = {}
        if self.session_token:
            headers["Authorization"] = f"Bearer {self.session_token}"
        return headers

    def get_health(self):
        try:
            res = requests.get(f"{self.base_url}/health", timeout=3.0)
            if res.status_code == 200:
                return {"status": "ok", "data": res.json()}
            return {"status": "error", "message": f"HTTP {res.status_code}"}
        except requests.exceptions.ConnectionError:
            return {"status": "disconnected", "message": "Connection refused"}
        except requests.exceptions.Timeout:
            return {"status": "timeout", "message": "Request timed out"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_models(self):
        try:
            res = requests.get(f"{self.base_url}/v1/models", timeout=3.0)
            if res.status_code == 200:
                return res.json()
            else:
                return {"error": f"HTTP {res.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    # Session auth handlers
    def signup(self, email, password):
        url = f"{self.base_url}/v1/auth/signup"
        try:
            res = requests.post(url, json={"email": email, "password": password}, timeout=5.0)
            return res.status_code, res.json()
        except Exception as e:
            return 500, {"error": str(e)}

    def login(self, email, password):
        url = f"{self.base_url}/v1/auth/login"
        try:
            res = requests.post(url, json={"email": email, "password": password}, timeout=5.0)
            return res.status_code, res.json()
        except Exception as e:
            return 500, {"error": str(e)}

    def logout(self):
        url = f"{self.base_url}/v1/auth/logout"
        headers = self._get_session_headers()
        try:
            res = requests.post(url, headers=headers, timeout=5.0)
            return res.status_code, res.json()
        except Exception as e:
            return 500, {"error": str(e)}

    # Developer actions (Securely wrapped with session headers)
    def generate_api_key(self, developer_id):
        url = f"{self.base_url}/v1/keys"
        headers = self._get_session_headers()
        try:
            res = requests.post(url, json={"developer_id": developer_id}, headers=headers, timeout=5.0)
            return res.status_code, res.json()
        except Exception as e:
            return 500, {"error": str(e)}

    def list_api_keys(self, developer_id):
        url = f"{self.base_url}/v1/developers/{developer_id}/keys"
        headers = self._get_session_headers()
        try:
            res = requests.get(url, headers=headers, timeout=5.0)
            return res.status_code, res.json()
        except Exception as e:
            return 500, {"error": str(e)}

    def revoke_api_key(self, key_id):
        url = f"{self.base_url}/v1/keys/{key_id}/revoke"
        headers = self._get_session_headers()
        try:
            res = requests.post(url, headers=headers, timeout=5.0)
            return res.status_code, res.json()
        except Exception as e:
            return 500, {"error": str(e)}

    def get_usage_stats(self, developer_id):
        url = f"{self.base_url}/v1/developers/{developer_id}/usage"
        headers = self._get_session_headers()
        try:
            res = requests.get(url, headers=headers, timeout=5.0)
            return res.status_code, res.json()
        except Exception as e:
            return 500, {"error": str(e)}

    # Generation completion call (uses API authorization key)
    def generate(self, prompt, model="collision-10m", max_tokens=100, temp=0.7, top_k=50, top_p=0.9):
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temp,
            "top_k": top_k,
            "top_p": top_p
        }
        
        url = f"{self.base_url}/v1/generate"
        headers = self._get_api_headers()
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=20.0)
            if res.status_code == 200:
                return {"success": True, "data": res.json(), "request_json": payload}
            else:
                try:
                    detail = res.json().get("error", {}).get("message", res.json().get("detail", res.text))
                except Exception:
                    detail = res.text
                return {"success": False, "error": f"HTTP {res.status_code}: {detail}", "request_json": payload}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Connection refused. FastAPI server is not running.", "request_json": payload}
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timed out.", "request_json": payload}
        except Exception as e:
            return {"success": False, "error": str(e), "request_json": payload}

    def submit_feedback(self, prompt, response, rating, category="general", feedback="", consent=True, model="collision-10m", user_id="anonymous"):
        payload = {
            "user_id": user_id,
            "prompt": prompt,
            "model": model,
            "response": response,
            "rating": rating,
            "feedback": feedback,
            "category": category,
            "consent": consent
        }
        url = f"{self.base_url}/v1/feedback"
        headers = self._get_api_headers()
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=5.0)
            return res.status_code, res.json()
        except Exception as e:
            return 500, {"error": str(e)}

