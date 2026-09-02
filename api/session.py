from fastapi import Header, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from api.database import verify_session

security_scheme = HTTPBearer(auto_error=False)

async def get_current_session_developer(
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
        
    raw_token = authorization.credentials
    if not raw_token.startswith("sess_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "type": "authentication_error",
                "message": "Invalid session token."
            }
        )
        
    dev_info = verify_session(raw_token)
    if not dev_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "type": "authentication_error",
                "message": "Session has expired or is invalid. Please log in again."
            }
        )
        
    return dev_info
