# COLLISION LAB Developer Playground

`COLLISION LAB` is a local Streamlit-based web playground for developers to interact with the frozen `COLLISION-10M` base language model through the FastAPI REST API.

## Architecture
The application runs as a clean decoupled model-view-controller pipeline:
```
User -> Streamlit (COLLISION LAB) -> FastAPI -> Inference Engine -> COLLISION-10M
```
The Streamlit app communicates with the model exclusively via the REST API endpoints and does not load weights directly.

## Installation & Setup

1. **Start the FastAPI backend server**:
   From the project directory:
   ```bash
   uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```
2. **Start the Streamlit playground**:
   From a separate terminal window:
   ```bash
   streamlit run playground/app.py
   ```

## Configuration
* **API base URL**: The URL is configurable in the sidebar. By default, it connects to `http://localhost:8000`. You can override it by setting the `COLLISION_API_URL` environment variable.

## Interface Sections
1. **Header & Status**: Shows real-time connection status queried from `/health`.
2. **Model Info & Limitations**: Displays active model configuration.
3. **Prompt Area**: Text box for queries with pre-loaded example prompts.
4. **Generation Controls**: Sliders and inputs for Temperature, Top P, Top K, and Max Tokens.
5. **Completions Output Panel**: Main area displaying the generated continuation.
6. **cURL & JSON Introspection**: Shows generated cURL commands and raw API requests/responses.
7. **Session History**: Logs requests and completion parameters inside the current session.
