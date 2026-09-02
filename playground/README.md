# COLLISION LAB Playground

Developer playground for interacting with the COLLISION-10M base language model using Streamlit.

## Features
* **Status Monitoring**: Shows real-time connection status of the FastAPI backend.
* **Active Controls**: Slider and input controls for temperature, top P, top K, and max tokens.
* **Output Introspection**: Displays only continuation output, highlighting prompt and completions.
* **Performance Telemetry**: Displays token completion counts, latency, and throughput rates.
* **cURL & JSON Introspection**: Shows API request and response JSON structure and generates ready-to-use cURL commands for terminal integration.

## Usage
1. Make sure your FastAPI backend is running:
   ```bash
   uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```
2. Start the Streamlit playground:
   ```bash
   streamlit run playground/app.py
   ```
