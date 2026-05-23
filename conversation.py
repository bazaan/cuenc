"""Manejo de estado de conversacion con Redis."""

import json
import logging
import redis.asyncio as redis
from models import ConversationState
from config import REDIS_URL, CONVERSATION_TTL

log = logging.getLogger(__name__)

_pool: redis.Redis | None = None


async def get_redis() -> redis.Redis:
    global _pool
    if _pool is None:
        _pool = redis.from_url(REDIS_URL, decode_responses=True)
    return _pool


def _key(conversation_id: int) -> str:
    return f"docc:conv:{conversation_id}"


async def get_state(conversation_id: int) -> ConversationState | None:
    """Obtiene el estado de una conversacion."""
    r = await get_redis()
    raw = await r.get(_key(conversation_id))
    if raw:
        return ConversationState(**json.loads(raw))
    return None


async def save_state(state: ConversationState):
    """Guarda el estado con TTL."""
    r = await get_redis()
    await r.setex(
        _key(state.conversation_id),
        CONVERSATION_TTL,
        state.model_dump_json(),
    )


async def delete_state(conversation_id: int):
    """Limpia el estado (conversacion completada)."""
    r = await get_redis()
    await r.delete(_key(conversation_id))


async def add_message(state: ConversationState, role: str, content: str) -> ConversationState:
    """Agrega un mensaje al historial (mantiene ultimos 10)."""
    state.messages.append({"role": role, "content": content})
    if len(state.messages) > 10:
        state.messages = state.messages[-10:]
    await save_state(state)
    return state
