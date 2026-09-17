import os
import time
import uuid
import hashlib
import threading
from fastapi import APIRouter, Depends, HTTPException, status, Header, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional

from api.schemas import (
    GenerateRequest, 
    GenerateResponse, 
    HealthResponse, 
    ModelListResponse, 
    ModelInfo, 
    UsageInfo, 
    PerformanceInfo,
    SourceInfo,
    FeedbackRequest,
    FeedbackResponse,
    AskRequest,
    AskResponse,
    ReadyResponse,
    SourceProvenance
)
from collision.service import get_collision_service, CollisionService
from api.dependencies import get_inference_engine
from collision.inference.engine import CollisionInferenceEngine
from rag.pipeline import RAGPipeline
from rag.schemas import RAGRequest, SourceItem

from api.auth import get_authenticated_developer
from api.session import get_current_session_developer
from api.limiter import check_rate_limit
from api.database import (
    get_db_connection,
    create_developer_with_password,
    create_session,
    revoke_session,
    verify_password,
    create_api_key,
    revoke_api_key,
    get_all_keys_for_developer,
    get_developer_usage_stats,
    log_usage_event,
    get_developer_by_email,
    record_feedback_event
)

router = APIRouter()

# Conservative concurrent generation limit for CPU execution (max 5 concurrent requests)
GENERATION_SEMAPHORE = threading.Semaphore(5)

class SignupRequest(BaseModel):
    email: str = Field(..., example="dev@example.com")
    password: str = Field(..., min_length=8, example="securepassword123")

class LoginRequest(BaseModel):
    email: str = Field(..., example="dev@example.com")
    password: str = Field(..., example="securepassword123")

class AuthResponse(BaseModel):
    session_token: str
    developer_id: int
    email: str

class DeveloperCreate(BaseModel):
    email: str = Field(..., example="dev@example.com")

class DeveloperResponse(BaseModel):
    id: int
    email: str

class KeyCreateRequest(BaseModel):
    developer_id: int

class KeyCreateResponse(BaseModel):
    api_key: str
    id: int
    prefix: str

class KeyListResponse(BaseModel):
    id: int
    prefix: str
    created_at: str
    last_used_at: Optional[str]
    revoked_at: Optional[str]
    status: str

class UsageStatsResponse(BaseModel):
    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    avg_latency_ms: float

# Custom Model Details Response
class ExtendedModelInfo(BaseModel):
    id: str
    parameter_count: int
    context_length: int
    model_type: str
    capabilities: List[str]
    status: str

class ExtendedModelListResponse(BaseModel):
    data: List[ExtendedModelInfo]

def verify_key_ownership(developer_id: int, key_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    if hasattr(cursor, "execute"):
        from api.database import execute_query
        cursor = execute_query(conn, "SELECT developer_id FROM api_keys WHERE id = %s", (key_id,))
    row = cursor.fetchone()
    conn.close()
    if not row or row["developer_id"] != developer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"type": "authorization_error", "message": "Access denied to this API key."}
        )

# Auth routes
@router.post("/v1/auth/signup", response_model=DeveloperResponse)
def signup(req: SignupRequest):
    dev_id = create_developer_with_password(req.email, req.password)
    if dev_id == -1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"type": "validation_error", "message": "Email address already registered."}
        )
    return DeveloperResponse(id=dev_id, email=req.email)

@router.post("/v1/auth/login", response_model=AuthResponse)
def login(req: LoginRequest):
    dev = get_developer_by_email(req.email)
    if not dev or not dev["password_hash"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"type": "authentication_error", "message": "Invalid email or password."}
        )
        
    if not verify_password(req.password, dev["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"type": "authentication_error", "message": "Invalid email or password."}
        )
        
    if dev["status"] == "suspended":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"type": "authorization_error", "message": "Developer account has been suspended."}
        )
        
    session_token = create_session(dev["id"])
    return AuthResponse(session_token=session_token, developer_id=dev["id"], email=dev["email"])

@router.post("/v1/auth/logout")
def logout(
    authorization: HTTPAuthorizationCredentials = Security(HTTPBearer(auto_error=False))
):
    if authorization and authorization.scheme.lower() == "bearer":
        raw_token = authorization.credentials
        revoke_session(raw_token)
    return {"status": "success", "message": "Logged out successfully."}

def verify_admin_token(x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token")):
    admin_secret = os.environ.get("ADMIN_SECRET")
    if not admin_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"type": "server_error", "message": "Admin authentication is not configured on server."}
        )
    if not x_admin_token or x_admin_token != admin_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"type": "authentication_error", "message": "Unauthorized. Invalid or missing X-Admin-Token."}
        )

@router.post("/v1/developers", response_model=DeveloperResponse)
def register_developer(req: DeveloperCreate, _admin = Depends(verify_admin_token)):
    conn = get_db_connection()
    try:
        from api.database import execute_query
        cursor = execute_query(conn, "INSERT INTO developers (email, status) VALUES (%s, 'active')", (req.email.strip().lower(),))
        conn.commit()
        from api.database import is_postgresql
        if is_postgresql():
            cursor = execute_query(conn, "SELECT id FROM developers WHERE email = %s", (req.email.strip().lower(),))
            row = cursor.fetchone()
            dev_id = row["id"] if row else -1
        else:
            dev_id = cursor.lastrowid
    except Exception:
        from api.database import execute_query
        cursor = execute_query(conn, "SELECT id FROM developers WHERE email = %s", (req.email.strip().lower(),))
        row = cursor.fetchone()
        dev_id = row["id"] if row else -1
    finally:
        conn.close()
    if dev_id == -1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"type": "validation_error", "message": "Failed to register developer."}
        )
    return DeveloperResponse(id=dev_id, email=req.email)

@router.get("/v1/developers/{email}", response_model=DeveloperResponse)
def get_developer(email: str, _admin = Depends(verify_admin_token)):
    dev = get_developer_by_email(email)
    if not dev:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"type": "validation_error", "message": "Developer not found."}
        )
    return DeveloperResponse(id=dev["id"], email=dev["email"])

# Secure developer portals
@router.post("/v1/keys", response_model=KeyCreateResponse)
def generate_key(req: KeyCreateRequest, current_dev: dict = Depends(get_current_session_developer)):
    if current_dev["developer_id"] != req.developer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"type": "authorization_error", "message": "Forbidden. Developer ID mismatch."}
        )
    try:
        raw_key, key_id = create_api_key(req.developer_id)
        return KeyCreateResponse(api_key=raw_key, id=key_id, prefix=raw_key[:8])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"type": "server_error", "message": f"Failed to generate API key: {str(e)}"}
        )

@router.get("/v1/developers/{developer_id}/keys", response_model=List[KeyListResponse])
def list_keys(developer_id: int, current_dev: dict = Depends(get_current_session_developer)):
    if current_dev["developer_id"] != developer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"type": "authorization_error", "message": "Forbidden. Developer ID mismatch."}
        )
    keys = get_all_keys_for_developer(developer_id)
    return [
        KeyListResponse(
            id=k["id"],
            prefix=k["key_prefix"],
            created_at=k["created_at"],
            last_used_at=k["last_used_at"],
            revoked_at=k["revoked_at"],
            status=k["status"]
        ) for k in keys
    ]

@router.post("/v1/keys/{key_id}/revoke")
def revoke_key(key_id: int, current_dev: dict = Depends(get_current_session_developer)):
    verify_key_ownership(current_dev["developer_id"], key_id)
    try:
        revoke_api_key(key_id)
        return {"status": "success", "message": "API key revoked successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"type": "server_error", "message": f"Failed to revoke API key: {str(e)}"}
        )

@router.get("/v1/developers/{developer_id}/usage", response_model=UsageStatsResponse)
def get_usage(developer_id: int, current_dev: dict = Depends(get_current_session_developer)):
    if current_dev["developer_id"] != developer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"type": "authorization_error", "message": "Forbidden. Developer ID mismatch."}
        )
    stats = get_developer_usage_stats(developer_id)
    return UsageStatsResponse(**stats)


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        service="collision",
        version="1.0",
        model="collision-10m",
        device="cpu"
    )

@router.get("/ready", response_model=ReadyResponse)
def ready(service: CollisionService = Depends(get_collision_service)):
    return service.ready()

@router.get("/metrics")
@router.get("/v1/metrics")
def get_metrics():
    from api.metrics import metrics_collector
    return metrics_collector.get_metrics()

@router.post("/v1/ask", response_model=AskResponse)
@router.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    http_req: Request,
    service: CollisionService = Depends(get_collision_service)
):
    from collision.config import COLLISION_MAX_INPUT_LENGTH, COLLISION_RATE_LIMIT_ENABLED
    
    # 1. Rate Limiting Protection
    if COLLISION_RATE_LIMIT_ENABLED:
        client_host = http_req.client.host if (http_req and http_req.client) else "127.0.0.1"
        rate_key = http_req.headers.get("X-API-Key", http_req.headers.get("X-Forwarded-For", client_host))
        check_rate_limit(rate_key)

    # 2. Input Length Enforcement
    if len(req.question) > COLLISION_MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "type": "validation_error",
                "message": f"Question length ({len(req.question)}) exceeds maximum limit of {COLLISION_MAX_INPUT_LENGTH} characters."
            }
        )

    # 3. Grounded Answering Service Execution
    res = service.ask(
        question=req.question,
        mode=req.mode or "AUTO",
        include_sources=req.include_sources if req.include_sources is not None else True,
        include_claims=req.include_claims if req.include_claims is not None else True
    )

    # Record state for structured logging & metrics
    http_req.state.mode = res.get("mode")
    http_req.state.answer_status = res.get("status")

    if res.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=res.get("error", {"code": "INVALID_REQUEST", "message": "Failed to process request."})
        )
    return res

@router.get("/v1/models", response_model=ExtendedModelListResponse)
def get_models():
    return ExtendedModelListResponse(
        data=[
            ExtendedModelInfo(
                id="collision-10m",
                parameter_count=10282304,
                context_length=256,
                model_type="base_language_model",
                capabilities=["text_completion"],
                status="operational"
            )
        ]
    )

@router.post("/v1/generate", response_model=GenerateResponse)
def generate(
    request: GenerateRequest, 
    http_req: Request,
    engine: CollisionInferenceEngine = Depends(get_inference_engine),
    developer: dict = Depends(get_authenticated_developer)
):
    # 1. Enforce rate limit
    check_rate_limit(developer["api_key_id"])

    # 2. Unknown model check
    if request.model != "collision-10m":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "type": "validation_error",
                "message": f"Model '{request.model}' is not supported. Available models: 'collision-10m'."
            }
        )

    # 3. Context Length and Token Bound Checks (HTTP 413)
    try:
        raw_prompt_tokens = len(engine.tokenizer.encode(request.prompt))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"type": "validation_error", "message": f"Tokenization failed: {str(e)}"}
        )
        
    if raw_prompt_tokens > 256:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"type": "validation_error", "message": f"Prompt length of {raw_prompt_tokens} tokens exceeds max context limit of 256."}
        )
        
    if raw_prompt_tokens + request.max_tokens > 256:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"type": "validation_error", "message": f"Combined prompt size ({raw_prompt_tokens}) and max_tokens ({request.max_tokens}) exceeds maximum context length of 256."}
        )

    # 4. Process Universal Hybrid RAG Pipeline
    sources_output = []
    rag_pipe = RAGPipeline(tokenizer=engine.tokenizer, inference_engine=engine)
    
    acquired = GENERATION_SEMAPHORE.acquire(timeout=5.0)
    if not acquired:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"type": "server_error", "message": "Server busy. Too many concurrent generation requests."}
        )

    t0 = time.perf_counter()
    try:
        rag_res = rag_pipe.process(
            RAGRequest(
                query=request.prompt,
                mode=request.web_search or "auto",
                top_k=3,
                max_tokens=request.max_tokens
            ),
            engine=engine
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"type": "model_error", "message": f"Generation failed: {str(e)}"}
        )
    finally:
        GENERATION_SEMAPHORE.release()

    latency_ms = (time.perf_counter() - t0) * 1000.0
    generated_text = rag_res.generated_answer or ""
    web_search_used = rag_res.web_search_used
    sources_output = [
        SourceInfo(title=s.title, url=s.url, snippet=s.snippet)
        for s in rag_res.sources
    ]

    try:
        prompt_tokens = len(engine.tokenizer.encode(request.prompt))
        completion_tokens = len(engine.tokenizer.encode(generated_text))
    except Exception:
        prompt_tokens = raw_prompt_tokens
        completion_tokens = max(1, len(generated_text.split()))
    total_tokens = prompt_tokens + completion_tokens

    # 5. Log usage events
    try:
        log_usage_event(
            developer_id=developer["developer_id"],
            api_key_id=developer["api_key_id"],
            model=request.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms
        )
    except Exception as e:
        print(f"Error logging usage event: {e}")
        
    # 6. Bind details to request.state for structured application logger
    http_req.state.model = request.model
    http_req.state.prompt_tokens = prompt_tokens
    http_req.state.completion_tokens = completion_tokens
    http_req.state.total_tokens = total_tokens
        
    return GenerateResponse(
        id=f"collision-generation-{uuid.uuid4()}",
        model=request.model,
        text=generated_text,
        web_search_used=web_search_used,
        sources=sources_output,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        ),
        performance=PerformanceInfo(
            latency_ms=latency_ms,
            tokens_per_second=completion_tokens / max(0.0001, latency_ms / 1000.0)
        )
    )

@router.post("/v1/playground/generate", response_model=GenerateResponse)
def playground_generate(
    request: GenerateRequest, 
    http_req: Request,
    engine: CollisionInferenceEngine = Depends(get_inference_engine),
    developer: dict = Depends(get_current_session_developer)
):
    # 1. Enforce rate limit
    developer_id = developer["developer_id"]
    keys = get_all_keys_for_developer(developer_id)
    active_keys = [k for k in keys if k["status"] == "active" and k["revoked_at"] is None]
    
    if active_keys:
        api_key_id = active_keys[0]["id"]
        rate_limit_key = api_key_id
    else:
        _, api_key_id = create_api_key(developer_id)
        rate_limit_key = f"dev_{developer_id}"
        
    check_rate_limit(rate_limit_key)

    # 2. Unknown model check
    if request.model != "collision-10m":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "type": "validation_error",
                "message": f"Model '{request.model}' is not supported. Available models: 'collision-10m'."
            }
        )

    # 3. Context Length and Token Bound Checks (HTTP 413)
    try:
        raw_prompt_tokens = len(engine.tokenizer.encode(request.prompt))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"type": "validation_error", "message": f"Tokenization failed: {str(e)}"}
        )
        
    if raw_prompt_tokens > 256:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"type": "validation_error", "message": f"Prompt length of {raw_prompt_tokens} tokens exceeds max context limit of 256."}
        )
        
    if raw_prompt_tokens + request.max_tokens > 256:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"type": "validation_error", "message": f"Combined prompt size ({raw_prompt_tokens}) and max_tokens ({request.max_tokens}) exceeds maximum context length of 256."}
        )

    # 4. Limit concurrency using thread-pool semaphore
    acquired = GENERATION_SEMAPHORE.acquire(timeout=5.0)
    if not acquired:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"type": "server_error", "message": "Server busy. Too many concurrent generation requests."}
        )

    t0 = time.perf_counter()
    rag_pipe = RAGPipeline(tokenizer=engine.tokenizer, inference_engine=engine)
    try:
        rag_res = rag_pipe.process(
            RAGRequest(
                query=request.prompt,
                mode=request.web_search or "auto",
                top_k=3,
                max_tokens=request.max_tokens
            ),
            engine=engine
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"type": "model_error", "message": f"Generation failed: {str(e)}"}
        )
    finally:
        GENERATION_SEMAPHORE.release()
    
    latency_ms = (time.perf_counter() - t0) * 1000.0
    generated_text = rag_res.generated_answer or ""
    web_search_used = rag_res.web_search_used
    sources_output = [
        SourceInfo(title=s.title, url=s.url, snippet=s.snippet)
        for s in rag_res.sources
    ]

    try:
        prompt_tokens = len(engine.tokenizer.encode(request.prompt))
        completion_tokens = len(engine.tokenizer.encode(generated_text))
    except Exception:
        prompt_tokens = raw_prompt_tokens
        completion_tokens = max(1, len(generated_text.split()))
    total_tokens = prompt_tokens + completion_tokens

    # 5. Log usage events
    try:
        log_usage_event(
            developer_id=developer_id,
            api_key_id=api_key_id,
            model=request.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms
        )
    except Exception as e:
        print(f"Error logging usage event: {e}")
        
    http_req.state.model = request.model
    http_req.state.prompt_tokens = prompt_tokens
    http_req.state.completion_tokens = completion_tokens
    http_req.state.total_tokens = total_tokens
        
    return GenerateResponse(
        id=f"collision-generation-{uuid.uuid4()}",
        model=request.model,
        text=generated_text,
        web_search_used=web_search_used,
        sources=sources_output,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        ),
        performance=PerformanceInfo(
            latency_ms=latency_ms,
            tokens_per_second=completion_tokens / max(0.0001, latency_ms / 1000.0)
        )
    )

@router.post("/v1/chat/generate", response_model=GenerateResponse)
def chat_generate(
    request: GenerateRequest, 
    http_req: Request,
    engine: CollisionInferenceEngine = Depends(get_inference_engine),
):
    # 1. Basic Rate limit check by client IP
    client_ip = http_req.client.host if http_req.client else "unknown_client"
    check_rate_limit(f"chat_{client_ip}")

    # 2. Unknown model check
    if request.model != "collision-10m":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "type": "validation_error",
                "message": f"Model '{request.model}' is not supported. Available models: 'collision-10m'."
            }
        )

    # 3. Context Length and Token Bound Checks (HTTP 413)
    try:
        raw_prompt_tokens = len(engine.tokenizer.encode(request.prompt))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"type": "validation_error", "message": f"Tokenization failed: {str(e)}"}
        )
        
    if raw_prompt_tokens > 256:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"type": "validation_error", "message": f"Prompt length of {raw_prompt_tokens} tokens exceeds max context limit of 256."}
        )
        
    if raw_prompt_tokens + request.max_tokens > 256:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"type": "validation_error", "message": f"Combined prompt size ({raw_prompt_tokens}) and max_tokens ({request.max_tokens}) exceeds maximum context length of 256."}
        )

    # 4. Limit concurrency using thread-pool semaphore
    acquired = GENERATION_SEMAPHORE.acquire(timeout=5.0)
    if not acquired:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"type": "server_error", "message": "Server busy. Too many concurrent generation requests."}
        )

    t0 = time.perf_counter()
    rag_pipe = RAGPipeline(tokenizer=engine.tokenizer, inference_engine=engine)
    try:
        rag_res = rag_pipe.process(
            RAGRequest(
                query=request.prompt,
                mode=request.web_search or "auto",
                top_k=3,
                max_tokens=request.max_tokens
            ),
            engine=engine
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"type": "model_error", "message": f"Generation failed: {str(e)}"}
        )
    finally:
        GENERATION_SEMAPHORE.release()
    
    latency_ms = (time.perf_counter() - t0) * 1000.0
    generated_text = rag_res.generated_answer or ""
    web_search_used = rag_res.web_search_used
    sources_output = [
        SourceInfo(title=s.title, url=s.url, snippet=s.snippet)
        for s in rag_res.sources
    ]

    try:
        prompt_tokens = len(engine.tokenizer.encode(request.prompt))
        completion_tokens = len(engine.tokenizer.encode(generated_text))
    except Exception:
        prompt_tokens = raw_prompt_tokens
        completion_tokens = max(1, len(generated_text.split()))
    total_tokens = prompt_tokens + completion_tokens

    http_req.state.model = request.model
    http_req.state.prompt_tokens = prompt_tokens
    http_req.state.completion_tokens = completion_tokens
    http_req.state.total_tokens = total_tokens
        
    return GenerateResponse(
        id=f"collision-generation-{uuid.uuid4()}",
        model=request.model,
        text=generated_text,
        web_search_used=web_search_used,
        sources=sources_output,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        ),
        performance=PerformanceInfo(
            latency_ms=latency_ms,
            tokens_per_second=completion_tokens / max(0.0001, latency_ms / 1000.0)
        )
    )

@router.post("/v1/ask", response_model=AskResponse)
def ask_question(
    req: AskRequest,
    engine: CollisionInferenceEngine = Depends(get_inference_engine)
):
    """
    Universal question answering endpoint answering from greetings ('hi') to complex open-domain queries.
    """
    t0 = time.perf_counter()
    rag_pipe = RAGPipeline(tokenizer=engine.tokenizer, inference_engine=engine)
    rag_res = rag_pipe.process_universal(
        query=req.question,
        mode=req.mode.lower() if req.mode else "auto",
        engine=engine
    )
    total_latency = (time.perf_counter() - t0) * 1000.0

    sources = [
        SourceProvenance(
            source_id=f"src-{idx+1}",
            title=s.title,
            url=s.url,
            source_type="WEB" if rag_res.web_search_used else "LOCAL",
            snippet=s.snippet
        )
        for idx, s in enumerate(rag_res.sources)
    ] if req.include_sources else []

    from api.schemas import LatencyBreakdown
    return AskResponse(
        answer=rag_res.generated_answer or "No answer could be generated.",
        status="ANSWERED",
        mode=rag_res.mode_used.upper(),
        confidence=0.95 if (rag_res.generated_answer and not rag_res.error) else 0.50,
        sources=sources,
        claims=[],
        latency=LatencyBreakdown(
            routing_ms=rag_res.search_latency_ms,
            retrieval_ms=rag_res.fetch_latency_ms + rag_res.rank_latency_ms,
            generation_ms=max(0.0, total_latency - rag_res.total_rag_latency_ms),
            total_ms=total_latency
        ),
        metadata={"web_search_used": rag_res.web_search_used}
    )

@router.post("/v1/feedback", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest):
    try:
        fb_id = record_feedback_event(
            user_id=req.user_id,
            prompt=req.prompt,
            model=req.model,
            response=req.response,
            rating=req.rating,
            feedback=req.feedback,
            category=req.category,
            consent=req.consent
        )
        return FeedbackResponse(id=fb_id, status="success", message="Feedback recorded successfully.")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"type": "server_error", "message": f"Failed to record feedback: {str(e)}"}
        )


