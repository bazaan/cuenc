"""Autenticacion JWT + gestion de usuarios."""

import logging
from datetime import datetime, timedelta, timezone
import jwt
import bcrypt
import asyncpg
from config import JWT_SECRET, JWT_EXPIRE_HOURS

log = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_token(user_id: int, username: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


async def init_users_table(pool: asyncpg.Pool):
    """Crea tabla de usuarios y usuario admin por defecto."""
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS usuarios (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                nombre TEXT NOT NULL,
                role TEXT DEFAULT 'staff',
                activo BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        # Crear admin si no existe
        exists = await conn.fetchval(
            "SELECT COUNT(*) FROM usuarios WHERE username = 'admin'"
        )
        if not exists:
            pw = hash_password("respiravida2024")
            await conn.execute(
                "INSERT INTO usuarios (username, password_hash, nombre, role) VALUES ($1, $2, $3, $4)",
                "admin", pw, "Administrador", "admin",
            )
            log.info("Usuario admin creado (pass: respiravida2024)")


async def authenticate(pool: asyncpg.Pool, username: str, password: str) -> dict | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM usuarios WHERE username = $1 AND activo = TRUE", username
        )
        if row and verify_password(password, row["password_hash"]):
            return dict(row)
    return None


async def get_users(pool: asyncpg.Pool) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, username, nombre, role, activo, created_at FROM usuarios ORDER BY id"
        )
        return [dict(r) for r in rows]


async def create_user(pool: asyncpg.Pool, username: str, password: str, nombre: str, role: str = "staff") -> int:
    async with pool.acquire() as conn:
        pw = hash_password(password)
        row = await conn.fetchrow(
            "INSERT INTO usuarios (username, password_hash, nombre, role) VALUES ($1, $2, $3, $4) RETURNING id",
            username, pw, nombre, role,
        )
        return row["id"]


async def delete_user(pool: asyncpg.Pool, user_id: int):
    async with pool.acquire() as conn:
        await conn.execute("UPDATE usuarios SET activo = FALSE WHERE id = $1", user_id)


async def find_or_create_google_user(pool: asyncpg.Pool, email: str, name: str) -> dict:
    """Busca usuario por email o lo crea automaticamente (login con Google)."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, username, nombre, role FROM usuarios WHERE username = $1 AND activo = TRUE",
            email,
        )
        if row:
            return dict(row)
        # Crear usuario nuevo con password aleatorio (solo usara Google login)
        import secrets
        pwd_hash = hash_password(secrets.token_hex(16))
        row = await conn.fetchrow(
            "INSERT INTO usuarios (username, password_hash, nombre, role, activo) "
            "VALUES ($1, $2, $3, 'staff', TRUE) RETURNING id, username, nombre, role",
            email, pwd_hash, name or email.split("@")[0],
        )
        log.info(f"Usuario Google creado: {email} ({name})")
        return dict(row)
