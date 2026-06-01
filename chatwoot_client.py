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


async def remove_label(conversation_id: int, label: str):
    """Remueve un label de la conversacion."""
    url = f"{BASE}/conversations/{conversation_id}/labels"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, headers=HEADERS)
        current = resp.json().get("payload", []) if resp.status_code < 400 else []
        if label in current:
            current.remove(label)
            await client.post(url, json={"labels": current}, headers=HEADERS)


async def get_conversation_labels(conversation_id: int) -> list:
    """Obtiene los labels actuales de una conversacion."""
    url = f"{BASE}/conversations/{conversation_id}/labels"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, headers=HEADERS)
        if resp.status_code < 400:
            return resp.json().get("payload", [])
        return []


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


async def set_custom_attributes(conversation_id: int, attrs: dict):
    """Actualiza custom attributes en una conversacion (merge con existentes)."""
    url_conv = f"{BASE}/conversations/{conversation_id}"
    url_attrs = f"{BASE}/conversations/{conversation_id}/custom_attributes"
    async with httpx.AsyncClient(timeout=10) as client:
        # Leer atributos actuales
        resp = await client.get(url_conv, headers=HEADERS)
        current = {}
        if resp.status_code < 400:
            current = resp.json().get("custom_attributes", {})
        # Merge con nuevos
        current.update(attrs)
        resp = await client.post(url_attrs, json={"custom_attributes": current}, headers=HEADERS)
        if resp.status_code >= 400:
            log.error(f"Chatwoot set_custom_attributes error {resp.status_code}: {resp.text}")
            return False
        return True


async def get_custom_attributes(conversation_id: int) -> dict:
    """Lee los custom attributes de una conversacion desde Chatwoot."""
    url = f"{BASE}/conversations/{conversation_id}"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, headers=HEADERS)
        if resp.status_code >= 400:
            return {}
        return resp.json().get("custom_attributes", {})


async def ensure_account_labels(labels: list[dict]):
    """Crea labels a nivel de cuenta si no existen. Cada dict: {title, description, color}."""
    url = f"{BASE}/labels"
    async with httpx.AsyncClient(timeout=10) as client:
        # Obtener labels existentes
        resp = await client.get(url, headers=HEADERS)
        existing = set()
        if resp.status_code < 400:
            for lbl in resp.json().get("payload", []):
                existing.add(lbl.get("title", ""))

        for lbl in labels:
            if lbl["title"] not in existing:
                resp = await client.post(url, json=lbl, headers=HEADERS)
                if resp.status_code < 400:
                    log.info(f"Label '{lbl['title']}' creado en Chatwoot")
                else:
                    log.warning(f"No se pudo crear label '{lbl['title']}': {resp.status_code} {resp.text}")


async def list_open_conversations(page: int = 1) -> list[dict]:
    """Lista conversaciones abiertas (paginadas, 25 por página)."""
    url = f"{BASE}/conversations?status=open&page={page}"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers=HEADERS)
        if resp.status_code >= 400:
            log.error(f"Chatwoot list conversations error {resp.status_code}")
            return []
        data = resp.json().get("data", {})
        return data.get("payload", [])


async def get_conversation_messages(conversation_id: int, limit: int = 5) -> list[dict]:
    """Obtiene los últimos mensajes de una conversación (más reciente primero)."""
    url = f"{BASE}/conversations/{conversation_id}/messages"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, headers=HEADERS)
        if resp.status_code >= 400:
            return []
        msgs = resp.json().get("payload", [])
        if isinstance(msgs, dict):
            msgs = msgs.get("messages", [])
        # Chatwoot retorna en orden ascendente — invertir para tener más reciente primero
        msgs.reverse()
        return msgs[:limit]


async def toggle_status(conversation_id: int, status: str = "open"):
    """Cambia el status de una conversacion (open, pending, resolved, snoozed)."""
    url = f"{BASE}/conversations/{conversation_id}/toggle_status"
    payload = {"status": status}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(url, json=payload, headers=HEADERS)
        if resp.status_code >= 400:
            log.error(f"Chatwoot toggle status error {resp.status_code}: {resp.text}")
        return resp.status_code < 400
