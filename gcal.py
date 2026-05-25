"""Google Calendar integration — OAuth2 + Calendar API."""

import asyncio
import json
import logging
from datetime import datetime, date, time, timedelta

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI

log = logging.getLogger(__name__)

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/calendar",
]


def _client_config():
    return {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [GOOGLE_REDIRECT_URI],
        }
    }


def _build_creds(token_data: dict) -> Credentials:
    return Credentials(
        token=token_data.get("token"),
        refresh_token=token_data.get("refresh_token"),
        token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=token_data.get("client_id", GOOGLE_CLIENT_ID),
        client_secret=token_data.get("client_secret", GOOGLE_CLIENT_SECRET),
        scopes=token_data.get("scopes", SCOPES),
    )


def _get_service(token_data: dict):
    creds = _build_creds(token_data)
    return build("calendar", "v3", credentials=creds)


# ── Sync helpers (run via asyncio.to_thread) ──

def _get_auth_url_sync(state: str) -> str:
    flow = Flow.from_client_config(_client_config(), scopes=SCOPES)
    flow.redirect_uri = GOOGLE_REDIRECT_URI
    url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state=state,
    )
    return url


def _exchange_code_sync(code: str) -> dict:
    flow = Flow.from_client_config(_client_config(), scopes=SCOPES)
    flow.redirect_uri = GOOGLE_REDIRECT_URI
    flow.fetch_token(code=code)
    creds = flow.credentials
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or []),
    }


def _get_calendar_email_sync(token_data: dict) -> str:
    service = _get_service(token_data)
    cal = service.calendarList().get(calendarId="primary").execute()
    return cal.get("id", "")


def _create_event_sync(token_data, nombre, telefono, fecha, hora, motivo, duracion_min):
    service = _get_service(token_data)
    start_dt = datetime.combine(fecha, hora)
    end_dt = start_dt + timedelta(minutes=duracion_min)
    event = {
        "summary": f"Cita: {nombre}",
        "description": f"Paciente: {nombre}\nTelefono: {telefono}\nMotivo: {motivo or 'Consulta'}",
        "start": {"dateTime": start_dt.isoformat(), "timeZone": "America/Lima"},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": "America/Lima"},
        "reminders": {"useDefault": False, "overrides": [{"method": "popup", "minutes": 30}]},
    }
    result = service.events().insert(calendarId="primary", body=event).execute()
    return result.get("id")


def _list_events_sync(token_data, fecha_inicio, fecha_fin):
    service = _get_service(token_data)
    time_min = datetime.combine(fecha_inicio, time(0, 0)).isoformat() + "-05:00"
    time_max = datetime.combine(fecha_fin + timedelta(days=1), time(0, 0)).isoformat() + "-05:00"
    result = service.events().list(
        calendarId="primary",
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy="startTime",
        maxResults=200,
    ).execute()
    events = []
    for ev in result.get("items", []):
        start = ev.get("start", {})
        end = ev.get("end", {})
        events.append({
            "id": ev.get("id"),
            "summary": ev.get("summary", ""),
            "description": ev.get("description", ""),
            "start": start.get("dateTime", start.get("date", "")),
            "end": end.get("dateTime", end.get("date", "")),
            "all_day": "date" in start,
        })
    return events


def _get_busy_sync(token_data, fecha, hora_inicio, hora_fin):
    service = _get_service(token_data)
    start_dt = datetime.combine(fecha, hora_inicio)
    end_dt = datetime.combine(fecha, hora_fin)
    body = {
        "timeMin": start_dt.isoformat() + "-05:00",
        "timeMax": end_dt.isoformat() + "-05:00",
        "items": [{"id": "primary"}],
    }
    result = service.freebusy().query(body=body).execute()
    busy = result.get("calendars", {}).get("primary", {}).get("busy", [])
    busy_times = []
    for b in busy:
        s = datetime.fromisoformat(b["start"].replace("Z", "+00:00"))
        e = datetime.fromisoformat(b["end"].replace("Z", "+00:00"))
        busy_times.append((s, e))
    return busy_times


# Colores GCal por estado de cita
GCAL_COLORS = {
    "pendiente": "6",     # Tangerine (naranja) — pre-agenda
    "confirmada": "10",   # Basil (verde) — confirmada
    "atendida": "9",      # Blueberry (azul) — atendida
    "no_contesto": "8",   # Graphite (gris)
    "cancelada": None,    # Se elimina del calendario
}


def _update_event_sync(token_data, event_id, nombre=None, telefono=None,
                       fecha=None, hora=None, motivo=None, estado=None, duracion_min=15):
    """Actualiza un evento existente en Google Calendar."""
    service = _get_service(token_data)
    event = service.events().get(calendarId="primary", eventId=event_id).execute()

    if nombre is not None:
        event["summary"] = f"Cita: {nombre}"
    if any(x is not None for x in (nombre, telefono, motivo)):
        event["description"] = (
            f"Paciente: {nombre or event.get('summary','').replace('Cita: ','')}\n"
            f"Telefono: {telefono or ''}\n"
            f"Motivo: {motivo or 'Consulta'}"
        )
    if fecha is not None and hora is not None:
        start_dt = datetime.combine(fecha, hora)
        end_dt = start_dt + timedelta(minutes=duracion_min)
        event["start"] = {"dateTime": start_dt.isoformat(), "timeZone": "America/Lima"}
        event["end"] = {"dateTime": end_dt.isoformat(), "timeZone": "America/Lima"}
    if estado and GCAL_COLORS.get(estado):
        event["colorId"] = GCAL_COLORS[estado]

    result = service.events().update(calendarId="primary", eventId=event_id, body=event).execute()
    return result.get("id")


def _delete_event_sync(token_data, event_id):
    """Elimina un evento de Google Calendar."""
    service = _get_service(token_data)
    service.events().delete(calendarId="primary", eventId=event_id).execute()


def _get_user_info_sync(token_data: dict) -> dict:
    """Obtiene email y nombre del usuario Google usando el access token."""
    import httpx
    resp = httpx.get(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {token_data['token']}"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "email": data.get("email", ""),
        "name": data.get("name", ""),
        "picture": data.get("picture", ""),
    }


# ── Async wrappers ──

async def get_auth_url(state: str = "") -> str:
    return await asyncio.to_thread(_get_auth_url_sync, state)


async def get_user_info(token_data: dict) -> dict:
    try:
        return await asyncio.to_thread(_get_user_info_sync, token_data)
    except Exception as e:
        log.error(f"GCal get_user_info error: {e}")
        return {"email": "", "name": "", "picture": ""}


async def exchange_code(code: str) -> dict:
    return await asyncio.to_thread(_exchange_code_sync, code)


async def get_calendar_email(token_data: dict) -> str:
    try:
        return await asyncio.to_thread(_get_calendar_email_sync, token_data)
    except Exception as e:
        log.error(f"GCal get_email error: {e}")
        return ""


async def create_event(token_data, nombre, telefono, fecha, hora, motivo="",
                       estado="pendiente", duracion_min=15) -> str | None:
    try:
        eid = await asyncio.to_thread(
            _create_event_sync, token_data, nombre, telefono, fecha, hora, motivo, duracion_min
        )
        # Asignar color segun estado
        if eid and GCAL_COLORS.get(estado):
            try:
                await asyncio.to_thread(
                    _update_event_sync, token_data, eid, estado=estado,
                )
            except Exception:
                pass
        log.info(f"GCal event created: {eid} (estado={estado})")
        return eid
    except Exception as e:
        log.error(f"GCal create_event error: {e}")
        return None


async def update_event(token_data, event_id, **kwargs) -> bool:
    try:
        await asyncio.to_thread(_update_event_sync, token_data, event_id, **kwargs)
        log.info(f"GCal event updated: {event_id}")
        return True
    except Exception as e:
        log.error(f"GCal update_event error: {e}")
        return False


async def delete_event(token_data, event_id) -> bool:
    try:
        await asyncio.to_thread(_delete_event_sync, token_data, event_id)
        log.info(f"GCal event deleted: {event_id}")
        return True
    except Exception as e:
        log.error(f"GCal delete_event error: {e}")
        return False


async def list_events(token_data, fecha_inicio, fecha_fin) -> list[dict]:
    try:
        return await asyncio.to_thread(_list_events_sync, token_data, fecha_inicio, fecha_fin)
    except Exception as e:
        log.error(f"GCal list_events error: {e}")
        return []


async def get_busy_slots(token_data, fecha, hora_inicio=time(8, 30), hora_fin=time(18, 0)):
    try:
        return await asyncio.to_thread(_get_busy_sync, token_data, fecha, hora_inicio, hora_fin)
    except Exception as e:
        log.error(f"GCal get_busy error: {e}")
        return []
