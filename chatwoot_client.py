"""Cliente para enviar mensajes via Chatwoot API."""

import httpx
import logging
from config import CHATWOOT_BASE_URL, CHATWOOT_API_TOKEN, CHATWOOT_ACCOUNT_ID

log = logging.getLogger(__name__)

BASE = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}"
HEADERS = {
    "api_access_token": CHATWOOT_API_TOKEN,
    "Content-Type": "application/json",
}


async def send_message(conversation_id: int, text: str, private: bool = False) -> dict:
    """Envia un mensaje a una conversacion de Chatwoot."""
    url = f"{BASE}/conversations/{conversation_id}/messages"
    payload = {
        "content": text,
        "message_type": "outgoing",
        "private": private,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, json=payload, headers=HEADERS)
        if resp.status_code >= 400:
            log.error(f"Chatwoot send error {resp.status_code}: {resp.text}")
            return {}
        return resp.json()


async def get_contact(contact_id: int) -> dict:
    """Obtiene datos de un contacto."""
    url = f"{BASE}/contacts/{contact_id}"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, headers=HEADERS)
        if resp.status_code >= 400:
            return {}
        return resp.json()


async def add_label(conversation_id: int, label: str):
    """Agrega un label a la conversacion (ej: 'cita_agendada')."""
    url = f"{BASE}/conversations/{conversation_id}/labels"
    async with httpx.AsyncClient(timeout=10) as client:
        # Get current labels first
        resp = await client.get(url, headers=HEADERS)
        current = resp.json().get("payload", []) if resp.status_code < 400 else []
        if label not in current:
            current.append(label)
            await client.post(url, json={"labels": current}, headers=HEADERS)


async def assign_team(conversation_id: int, team_id: int):
    """Asigna una conversacion a un team en Chatwoot."""
    url = f"{BASE}/conversations/{conversation_id}/assignments"
    payload = {"team_id": team_id}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(url, json=payload, headers=HEADERS)
        if resp.status_code >= 400:
            log.error(f"Chatwoot assign team error {resp.status_code}: {resp.text}")
            return False
        log.info(f"Conv {conversation_id} asignada a team {team_id}")
        return True


async def toggle_status(conversation_id: int, status: str = "open"):
    """Cambia el status de una conversacion (open, pending, resolved, snoozed)."""
    url = f"{BASE}/conversations/{conversation_id}/toggle_status"
    payload = {"status": status}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(url, json=payload, headers=HEADERS)
        if resp.status_code >= 400:
            log.error(f"Chatwoot toggle status error {resp.status_code}: {resp.text}")
        return resp.status_code < 400
