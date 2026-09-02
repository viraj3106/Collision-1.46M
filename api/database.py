import os
import sqlite3
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

DATABASE_URL = os.environ.get("DATABASE_URL", os.environ.get("COLLISION_DB_PATH", "collision_api.db"))

def is_postgresql() -> bool:
    return DATABASE_URL.startswith("postgres://") or DATABASE_URL.startswith("postgresql://")

def get_db_connection():
    if is_postgresql():
        import psycopg2
        from psycopg2.extras import RealDictConnection
        # Fix potential postgres:// to postgresql:// scheme issue in psycopg2
        url = DATABASE_URL
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        conn = psycopg2.connect(url, connection_factory=RealDictConnection)
        return conn
    else:
        # Fallback to local SQLite
        conn = sqlite3.connect(DATABASE_URL)
        conn.row_factory = sqlite3.Row
        return conn

def execute_query(conn, query: str, params: tuple = ()):
    cursor = conn.cursor()
    if not is_postgresql():
        # SQLite fallback: convert %s placeholder to ?
        query = query.replace("%s", "?")
    cursor.execute(query, params)
    return cursor

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if is_postgresql():
        # 1. Create developers
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS developers (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(50) NOT NULL CHECK(status IN ('active', 'suspended'))
        )
        """)
        
        # 2. Create api_keys
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id SERIAL PRIMARY KEY,
            developer_id INTEGER NOT NULL,
            key_hash VARCHAR(255) UNIQUE NOT NULL,
            key_prefix VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used_at TIMESTAMP,
            revoked_at TIMESTAMP,
            status VARCHAR(50) NOT NULL CHECK(status IN ('active', 'revoked')),
            FOREIGN KEY (developer_id) REFERENCES developers(id)
        )
        """)
        
        # 3. Create usage_events
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS usage_events (
            id SERIAL PRIMARY KEY,
            developer_id INTEGER NOT NULL,
            api_key_id INTEGER NOT NULL,
            model VARCHAR(255) NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            latency_ms REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (developer_id) REFERENCES developers(id),
            FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
        )
        """)
        
        # 4. Create sessions
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
            developer_id INTEGER NOT NULL,
            token_hash VARCHAR(255) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            revoked_at TIMESTAMP,
            FOREIGN KEY (developer_id) REFERENCES developers(id)
        )
        """)

        # 5. Create feedback
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255),
            prompt TEXT NOT NULL,
            model VARCHAR(255) NOT NULL,
            response TEXT NOT NULL,
            rating VARCHAR(50) NOT NULL,
            feedback TEXT,
            category VARCHAR(100),
            consent BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
    else:
        # SQLite Schema
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS developers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL CHECK(status IN ('active', 'suspended'))
        )
        """)
        # Check if password_hash column exists (migration check)
        cursor.execute("PRAGMA table_info(developers)")
        columns = [row["name"] for row in cursor.fetchall()]
        if "password_hash" not in columns:
            cursor.execute("ALTER TABLE developers ADD COLUMN password_hash TEXT")
            
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            developer_id INTEGER NOT NULL,
            key_hash TEXT UNIQUE NOT NULL,
            key_prefix TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used_at TIMESTAMP,
            revoked_at TIMESTAMP,
            status TEXT NOT NULL CHECK(status IN ('active', 'revoked')),
            FOREIGN KEY (developer_id) REFERENCES developers(id)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS usage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            developer_id INTEGER NOT NULL,
            api_key_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            latency_ms REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (developer_id) REFERENCES developers(id),
            FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            developer_id INTEGER NOT NULL,
            token_hash TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            revoked_at TIMESTAMP,
            FOREIGN KEY (developer_id) REFERENCES developers(id)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            prompt TEXT NOT NULL,
            model TEXT NOT NULL,
            response TEXT NOT NULL,
            rating TEXT NOT NULL,
            feedback TEXT,
            category TEXT,
            consent INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

# Cryptographic password utilities using PBKDF2 (standard library)
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    pw_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt.hex() + ":" + pw_hash.hex()

def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, hash_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        pw_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return secrets.compare_digest(pw_hash.hex(), hash_hex)
    except Exception:
        return False

def create_developer_with_password(email: str, password: str) -> int:
    password_hash = hash_password(password)
    conn = get_db_connection()
    try:
        cursor = execute_query(
            conn,
            "INSERT INTO developers (email, password_hash, status) VALUES (%s, %s, 'active')",
            (email.strip().lower(), password_hash)
        )
        conn.commit()
        if is_postgresql():
            # Postgres needs explicit RETURNING or query id
            cursor = execute_query(conn, "SELECT id FROM developers WHERE email = %s", (email.strip().lower(),))
            row = cursor.fetchone()
            dev_id = row["id"] if row else -1
        else:
            dev_id = cursor.lastrowid
    except Exception:
        dev_id = -1
    finally:
        conn.close()
    return dev_id

def create_developer(email: str) -> int:
    conn = get_db_connection()
    try:
        cursor = execute_query(
            conn,
            "INSERT INTO developers (email, status) VALUES (%s, 'active')",
            (email.strip().lower(),)
        )
        conn.commit()
        if is_postgresql():
            cursor = execute_query(conn, "SELECT id FROM developers WHERE email = %s", (email.strip().lower(),))
            row = cursor.fetchone()
            dev_id = row["id"] if row else -1
        else:
            dev_id = cursor.lastrowid
    except Exception:
        cursor = execute_query(conn, "SELECT id FROM developers WHERE email = %s", (email.strip().lower(),))
        row = cursor.fetchone()
        dev_id = row["id"] if row else -1
    finally:
        conn.close()
    return dev_id

def get_developer_by_email(email: str) -> Optional[Dict]:
    conn = get_db_connection()
    cursor = execute_query(conn, "SELECT * FROM developers WHERE email = %s", (email.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def generate_secure_api_key() -> str:
    secure_hex = secrets.token_hex(16)
    return f"col_{secure_hex}"

def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode('utf-8')).hexdigest()

def create_api_key(developer_id: int) -> Tuple[str, int]:
    raw_key = generate_secure_api_key()
    key_hash = hash_api_key(raw_key)
    key_prefix = raw_key[:8] # col_xxxx
    
    conn = get_db_connection()
    cursor = execute_query(
        conn,
        "INSERT INTO api_keys (developer_id, key_hash, key_prefix, status) VALUES (%s, %s, %s, 'active')",
        (developer_id, key_hash, key_prefix)
    )
    conn.commit()
    if is_postgresql():
        cursor = execute_query(conn, "SELECT id FROM api_keys WHERE key_hash = %s", (key_hash,))
        row = cursor.fetchone()
        key_id = row["id"] if row else -1
    else:
        key_id = cursor.lastrowid
    conn.close()
    return raw_key, key_id

def get_api_key_by_hash(key_hash: str) -> Optional[Dict]:
    conn = get_db_connection()
    cursor = execute_query(conn, """
        SELECT ak.*, dev.email as developer_email, dev.status as developer_status 
        FROM api_keys ak
        JOIN developers dev ON ak.developer_id = dev.id
        WHERE ak.key_hash = %s
    """, (key_hash,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def update_api_key_last_used(key_id: int):
    conn = get_db_connection()
    execute_query(
        conn,
        "UPDATE api_keys SET last_used_at = %s WHERE id = %s",
        (datetime.utcnow().isoformat(), key_id)
    )
    conn.commit()
    conn.close()

def revoke_api_key(key_id: int):
    conn = get_db_connection()
    execute_query(
        conn,
        "UPDATE api_keys SET status = 'revoked', revoked_at = %s WHERE id = %s",
        (datetime.utcnow().isoformat(), key_id)
    )
    conn.commit()
    conn.close()

def log_usage_event(developer_id: int, api_key_id: int, model: str, prompt_tokens: int, completion_tokens: int, latency_ms: float):
    conn = get_db_connection()
    execute_query(
        conn,
        "INSERT INTO usage_events (developer_id, api_key_id, model, prompt_tokens, completion_tokens, latency_ms) VALUES (%s, %s, %s, %s, %s, %s)",
        (developer_id, api_key_id, model, prompt_tokens, completion_tokens, latency_ms)
    )
    conn.commit()
    conn.close()

def get_all_keys_for_developer(developer_id: int) -> List[Dict]:
    conn = get_db_connection()
    cursor = execute_query(conn, "SELECT * FROM api_keys WHERE developer_id = %s", (developer_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_developer_usage_stats(developer_id: int) -> Dict:
    conn = get_db_connection()
    cursor = execute_query(conn, """
        SELECT 
            COUNT(*) as total_requests,
            SUM(prompt_tokens) as total_prompt_tokens,
            SUM(completion_tokens) as total_completion_tokens,
            AVG(latency_ms) as avg_latency_ms
        FROM usage_events
        WHERE developer_id = %s
    """, (developer_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row or row["total_requests"] == 0:
        return {
            "total_requests": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "avg_latency_ms": 0.0
        }
        
    prompt_tokens = row["total_prompt_tokens"] or 0
    completion_tokens = row["total_completion_tokens"] or 0
    
    return {
        "total_requests": row["total_requests"] or 0,
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "avg_latency_ms": row["avg_latency_ms"] or 0.0
    }

# Session Management Functions
def create_session(developer_id: int, expires_in_seconds: int = 3600) -> str:
    raw_token = f"sess_{secrets.token_hex(24)}"
    token_hash = hashlib.sha256(raw_token.encode('utf-8')).hexdigest()
    expires_at = (datetime.utcnow() + timedelta(seconds=expires_in_seconds)).isoformat()
    
    conn = get_db_connection()
    cursor = execute_query(
        conn,
        "INSERT INTO sessions (developer_id, token_hash, expires_at) VALUES (%s, %s, %s)",
        (developer_id, token_hash, expires_at)
    )
    conn.commit()
    conn.close()
    return raw_token

def verify_session(token: str) -> Optional[Dict]:
    if not token.startswith("sess_"):
        return None
        
    token_hash = hashlib.sha256(token.encode('utf-8')).hexdigest()
    conn = get_db_connection()
    cursor = execute_query(conn, """
        SELECT s.*, dev.email as developer_email, dev.status as developer_status 
        FROM sessions s
        JOIN developers dev ON s.developer_id = dev.id
        WHERE s.token_hash = %s AND s.revoked_at IS NULL
    """, (token_hash,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
        
    # Check expiration
    now = datetime.utcnow().isoformat()
    if row["expires_at"] < now:
        return None
        
    if row["developer_status"] == "suspended":
        return None
        
    return {
        "developer_id": row["developer_id"],
        "email": row["developer_email"]
    }

def revoke_session(token: str):
    token_hash = hashlib.sha256(token.encode('utf-8')).hexdigest()
    conn = get_db_connection()
    execute_query(
        conn,
        "UPDATE sessions SET revoked_at = %s WHERE token_hash = %s",
        (datetime.utcnow().isoformat(), token_hash)
    )
    conn.commit()
    conn.close()

def record_feedback_event(
    user_id: Optional[str],
    prompt: str,
    model: str,
    response: str,
    rating: str,
    feedback: Optional[str] = None,
    category: Optional[str] = None,
    consent: bool = True
) -> int:
    conn = get_db_connection()
    query = """
        INSERT INTO feedback (user_id, prompt, model, response, rating, feedback, category, consent)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    params = (user_id or "anonymous", prompt, model, response, rating, feedback, category, consent)
    cursor = execute_query(conn, query, params)
    conn.commit()
    
    if is_postgresql():
        cursor = execute_query(conn, "SELECT MAX(id) as id FROM feedback")
        row = cursor.fetchone()
        fb_id = row["id"] if row else 1
    else:
        fb_id = cursor.lastrowid or 1
        
    conn.close()
    return fb_id


# Migration check on import (suppress for tests/daemons depending on init workflow)
try:
    init_db()
except Exception as e:
    print(f"Migration / database setup initialization deferred: {e}")
