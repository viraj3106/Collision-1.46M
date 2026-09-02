import os
import time
from fastapi import HTTPException, status
from collections import defaultdict
from typing import Dict, List, Union

# Local in-memory fallback store
_local_request_store: Dict[int, List[float]] = defaultdict(list)

# Redis client placeholder
_redis_client = None
_redis_failed = False

def get_redis_client():
    global _redis_client, _redis_failed
    if _redis_failed:
        return None
    if _redis_client is None:
        redis_url = os.environ.get("REDIS_URL")
        if redis_url:
            try:
                import redis
                _redis_client = redis.Redis.from_url(redis_url, socket_timeout=1.0, decode_responses=True)
                _redis_client.ping()
                print("Successfully connected to Redis for rate limiting.")
            except Exception as e:
                print(f"Failed to connect to Redis at {redis_url}: {e}. Falling back to local in-memory limiting.")
                _redis_failed = True
                _redis_client = None
    return _redis_client

def clear_rate_limits():
    # Clear both Redis and in-memory store
    global _local_request_store, _redis_client, _redis_failed
    _local_request_store.clear()
    
    # Try clearing Redis if client exists
    client = get_redis_client()
    if client:
        try:
            # Delete keys matching pattern
            keys = client.keys("rate_limit:*")
            if keys:
                client.delete(*keys)
        except Exception as e:
            print(f"Failed to clear Redis rate limits: {e}")

def _check_in_memory_rate_limit(rate_limit_key: Union[int, str], rate_limit: int):
    now = time.time()
    one_minute_ago = now - 60.0
    
    # Filter request timestamps older than one minute
    timestamps = _local_request_store[rate_limit_key]
    timestamps = [t for t in timestamps if t > one_minute_ago]
    
    if len(timestamps) >= rate_limit:
        retry_after_seconds = int(60.0 - (now - timestamps[0])) if timestamps else 60
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(retry_after_seconds)},
            detail={
                "type": "rate_limit_error",
                "message": f"Rate limit of {rate_limit} requests/minute exceeded. Try again in {retry_after_seconds} seconds."
            }
        )
        
    timestamps.append(now)
    _local_request_store[rate_limit_key] = timestamps

def check_rate_limit(rate_limit_key: Union[int, str]):
    rate_limit = int(os.environ.get("COLLISION_RATE_LIMIT", "60"))
    if rate_limit <= 0:
        return
        
    client = get_redis_client()
    if not client:
        # Fallback to local in-memory store
        _check_in_memory_rate_limit(rate_limit_key, rate_limit)
        return

    now = time.time()
    key = f"rate_limit:{rate_limit_key}"
    one_minute_ago = now - 60.0
    
    try:
        # Atomic transaction using pipelined execution
        pipe = client.pipeline()
        
        # Prune old requests
        pipe.zremrangebyscore(key, 0, one_minute_ago)
        
        # Get count of requests in last 60 seconds
        pipe.zcard(key)
        
        # Retrieve the oldest request to calculate accurate Retry-After
        pipe.zrange(key, 0, 0, withscores=True)
        
        _, card, oldest = pipe.execute()
        
        if card >= rate_limit:
            oldest_ts = oldest[0][1] if oldest else now
            retry_after_seconds = int(60.0 - (now - oldest_ts))
            retry_after_seconds = max(1, retry_after_seconds)
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": str(retry_after_seconds)},
                detail={
                    "type": "rate_limit_error",
                    "message": f"Rate limit of {rate_limit} requests/minute exceeded. Try again in {retry_after_seconds} seconds."
                }
            )
            
        # Add current request with timestamp as both member and score
        pipe = client.pipeline()
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, 65) # Garbage collect keys automatically after 65 seconds
        pipe.execute()
        
    except HTTPException:
        # Re-raise HTTP rate limit exceptions
        raise
    except Exception as e:
        # Graceful failure: log connection failure once and fallback
        print(f"Redis operation error: {e}. Falling back to in-memory rate limiter.")
        _check_in_memory_rate_limit(rate_limit_key, rate_limit)

