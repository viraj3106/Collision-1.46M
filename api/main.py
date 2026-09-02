import os
import sys
import time
import uuid
import json
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.routes import router
from api.dependencies import get_inference_engine
from api.database import init_db, get_db_connection, is_postgresql
from api.limiter import get_redis_client

# Keep track of cold start time
cold_start_metrics = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI server starting up...")
    t0 = time.time()
    
    # Init DB tables
    init_db()
    
    # Trigger model load and warmup
    engine = get_inference_engine()
    
    elapsed = time.time() - t0
    cold_start_metrics["model_load_time_seconds"] = elapsed
    print(f"Server startup cold-start completed in {elapsed:.3f} seconds.")
    
    yield
    print("FastAPI server shutting down...")

app = FastAPI(
    title="COLLISION-10M REST API",
    description="Secure completions endpoint for the COLLISION-10M model.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
allowed_origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8501",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8501"
]

# Append PUBLIC_PORTAL_URL if configured
public_portal_url = os.environ.get("PUBLIC_PORTAL_URL")
if public_portal_url:
    allowed_origins.append(public_portal_url.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID & Structured Logger Middleware
@app.middleware("http")
async def add_request_id_and_headers(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = f"req_{uuid.uuid4().hex[:16]}"
    
    request.state.request_id = request_id
    
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{latency_ms / 1000.0:.4f}"
    
    # Structured application logging
    log_data = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": round(latency_ms, 2)
    }
    
    # Safely attach inference details from request.state (if set in generate route)
    model = getattr(request.state, "model", None)
    if model:
        log_data["model"] = model
        log_data["prompt_tokens"] = getattr(request.state, "prompt_tokens", 0)
        log_data["completion_tokens"] = getattr(request.state, "completion_tokens", 0)
        log_data["total_tokens"] = getattr(request.state, "total_tokens", 0)
        
    print(json.dumps(log_data))
    return response

# Custom HTTP Exception Handler to map structure format
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    request_id = getattr(request.state, "request_id", "req_unknown")
    
    # Safe check if detail is already a dict (our custom exceptions)
    detail = exc.detail
    if isinstance(detail, dict):
        error_type = detail.get("type", "server_error")
        error_message = detail.get("message", "An unexpected error occurred.")
    else:
        # Default fallback
        error_type = "server_error"
        if exc.status_code == 404:
            error_type = "validation_error"
            error_message = f"Path {request.url.path} not found."
        elif exc.status_code == 401:
            error_type = "authentication_error"
            error_message = str(detail)
        elif exc.status_code == 403:
            error_type = "authorization_error"
            error_message = str(detail)
        elif exc.status_code == 429:
            error_type = "rate_limit_error"
            error_message = str(detail)
        else:
            error_message = str(detail)
            
    headers = exc.headers if getattr(exc, "headers", None) else None
    return JSONResponse(
        status_code=exc.status_code,
        headers=headers,
        content={
            "error": {
                "type": error_type,
                "message": error_message,
                "request_id": request_id
            }
        }
    )

# Validation Error Handler (Pydantic models / validation exceptions)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "req_unknown")
    errors = exc.errors()
    error_msgs = []
    for err in errors:
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "invalid value")
        error_msgs.append(f"{loc}: {msg}")
    
    message = "; ".join(error_msgs) if error_msgs else "Validation failed."
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "type": "validation_error",
                "message": message,
                "request_id": request_id
            }
        }
    )

# General Catch-All Exception Handler to hide stack traces from clients
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "req_unknown")
    print(f"Server Error request_id={request_id}: {exc}", file=sys.stderr)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "type": "server_error",
                "message": "An internal server error occurred. Please contact administrator.",
                "request_id": request_id
            }
        }
    )

app.include_router(router)

@app.get("/coldstart")
def get_coldstart_metrics():
    return cold_start_metrics

# Readiness Probe
@app.get("/ready")
def readiness_probe():
    checks = {}
    
    # 1. Database Check
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        conn.close()
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {str(e)}"
        
    # 2. Redis Check (Optional/Required if configured)
    if os.environ.get("REDIS_URL"):
        try:
            client = get_redis_client()
            if client and client.ping():
                checks["redis"] = "ok"
            else:
                checks["redis"] = "disconnected"
        except Exception as e:
            checks["redis"] = f"error: {str(e)}"
            
    # 3. Model Engine Check
    try:
        engine = get_inference_engine()
        if engine:
            checks["model"] = "ok"
        else:
            checks["model"] = "uninitialized"
    except Exception as e:
        checks["model"] = f"error: {str(e)}"
        
    # Check overall health
    is_healthy = all(v == "ok" for k, v in checks.items() if k in ["database", "model"])
    if not is_healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unready", "checks": checks}
        )
        
    return {"status": "ready", "checks": checks}
