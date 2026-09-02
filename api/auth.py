import hashlib
from fastapi import Header, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from api.database import get_api_key_by_hash, update_api_key_last_used

security_scheme = HTTPBearer(auto_error=False)

def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode('utf-8')).hexdigest()

async def get_authenticated_developer(
    authorization: HTTPAuthorizationCredentials = Security(security_scheme)
) -> dict:
    if not authorization or authorization.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "type": "authentication_error",
                "message": "Missing or malformed Authorization header. Use Bearer scheme."
            }
        )
        
    raw_key = authorization.credentials
    if not raw_key.startswith("col_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "type": "authentication_error",
                "message": "Invalid API key format."
            }
        )
        
    # Hash raw key to verify against DB securely
    key_hash = hash_api_key(raw_key)
    key_record = get_api_key_by_hash(key_hash)
    
    if not key_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "type": "authentication_error",
                "message": "Invalid API key."
            }
        )
        
    if key_record["status"] == "revoked":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "type": "authentication_error",
                "message": "API key has been revoked."
            }
        )
        
    if key_record["developer_status"] == "suspended":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "type": "authentication_error",
                "message": "Developer account is suspended."
            }
        )
        
    # Update last used timestamp
    update_api_key_last_used(key_record["id"])
    
    return {
        "developer_id": key_record["developer_id"],
        "api_key_id": key_record["id"],
        "email": key_record["developer_email"]
    }
