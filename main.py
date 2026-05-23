"""
Agente IA — Clínica Respira Vida (Doc C)
FastAPI: webhook Chatwoot + API panel de citas
"""

import logging
import re
from datetime import date, datetime, timedelta
from contextlib import asynccontextmanager

from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import config
from models import ConversationState, Cita, EstadoCita, Canal
from conversation import get_state, save_state, delete_state, add_message
from ai_engine import generate_response, extract_appointment_json, clean_response
from chatwoot_client import send_message, add_label
import appointments

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("docc-agent")


# ─── Startup / Shutdown ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando agente Doc C...")
    await appointments.init_db()
    log.info(f"Agente listo en puerto {config.PORT}")
    yield
    log.info("Apagando agente...")


app = FastAPI(title="Doc C Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Utilidades ───

def detect_canal(inbox_channel_type: str) -> Canal:
    """Detecta el canal desde el tipo de inbox de Chatwoot."""
    mapping = {
        "Channel::Whatsapp": Canal.WHATSAPP,
        "Channel::Api": Canal.WHATSAPP,  # API channel suele ser WA
        "Channel::Instagram": Canal.INSTAGRAM,
        "Channel::FacebookPage": Canal.MESSENGER,
        "Channel::TikTok": Canal.TIKTOK,
    }
    return mapping.get(inbox_channel_type, Canal.WHATSAPP)


def parse_fecha_from_text(text: str) -> date | None:
    """Intenta extraer una fecha del texto del usuario."""
    hoy = date.today()
    text_lower = text.lower().strip()

    # Hoy / mañana
    if "hoy" in text_lower:
        return hoy
    if "mañana" in text_lower:
        return hoy + timedelta(days=1)

    # Dias de la semana
    dias = {
        "lunes": 0, "martes": 1, "miercoles": 2, "miércoles": 2,
        "jueves": 3, "viernes": 4, "sabado": 5, "sábado": 5,
    }
    for dia_nombre, dia_num in dias.items():
        if dia_nombre in text_lower:
            days_ahead = dia_num - hoy.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return hoy + timedelta(days=days_ahead)

    # Formato DD/MM o DD-MM
    match = re.search(r"(\d{1,2})[/-](\d{1,2})", text)
    if match:
        try:
            d, m = int(match.group(1)), int(match.group(2))
            return date(hoy.year, m, d)
        except ValueError:
            pass

    return None


def parse_hora_from_text(text: str) -> str | None:
    """Intenta extraer una hora del texto."""
    # Patrones: "9:00", "9am", "9:30 am", "10:00am", "2pm", "14:00"
    patterns = [
        r"(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)?",
        r"(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.)",
    ]

    text_lower = text.lower()
    for pat in patterns:
        match = re.search(pat, text_lower)
        if match:
            groups = match.groups()
            h = int(groups[0])
            m = int(groups[1]) if len(groups) > 2 and groups[1].isdigit() else 0
            period = groups[-1] if groups[-1] else None

            if period and ("p" in period) and h < 12:
                h += 12
            if period and ("a" in period) and h == 12:
                h = 0

            return f"{h:02d}:{m:02d}"

    return None


# ─── Webhook Chatwoot ───

@app.post("/webhook/chatwoot")
async def webhook_chatwoot(request: Request):
    """Recibe mensajes desde Chatwoot y responde con IA."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"ok": False}, status_code=400)

    event = payload.get("event")

    # Solo procesar mensajes entrantes
    if event != "message_created":
        return {"ok": True, "skipped": "not_message_created"}

    message = payload.get("content", "")
    message_type = payload.get("message_type")

    # Ignorar mensajes salientes (nuestros propios) y privados
    if message_type != "incoming":
        return {"ok": True, "skipped": "not_incoming"}

    if not message or not message.strip():
        return {"ok": True, "skipped": "empty_message"}

    # Extraer datos de la conversacion
    conversation = payload.get("conversation", {})
    conversation_id = conversation.get("id")
    inbox_id = conversation.get("inbox_id")
    channel_type = conversation.get("channel", {}).get("type", "")

    sender = payload.get("sender", {})
    contact_id = sender.get("id")
    contact_name = sender.get("name", "")
    contact_phone = sender.get("phone_number", "")

    if not conversation_id or not contact_id:
        return {"ok": True, "skipped": "no_conv_or_contact"}

    log.info(f"[Conv {conversation_id}] {contact_name}: {message[:80]}")

    # Obtener o crear estado de conversacion
    state = await get_state(conversation_id)
    if state is None:
        state = ConversationState(
            contact_id=contact_id,
            contact_name=contact_name or None,
            contact_phone=contact_phone or None,
            conversation_id=conversation_id,
            inbox_id=inbox_id,
            canal=detect_canal(channel_type),
        )

    # Actualizar datos de contacto si tenemos nuevos
    if contact_name and not state.contact_name:
        state.contact_name = contact_name
    if contact_phone and not state.contact_phone:
        state.contact_phone = contact_phone

    # Agregar mensaje del usuario al historial
    state = await add_message(state, "user", message)

    # Detectar fecha/hora en el mensaje del usuario
    fecha_detectada = parse_fecha_from_text(message)
    hora_detectada = parse_hora_from_text(message)

    if fecha_detectada:
        state.fecha_elegida = fecha_detectada.isoformat()
    if hora_detectada:
        state.hora_elegida = hora_detectada

    # Obtener slots disponibles si hay fecha
    slots = None
    fecha_ctx = None
    if state.fecha_elegida:
        fecha_obj = date.fromisoformat(state.fecha_elegida)
        # No ofrecer domingos
        if fecha_obj.weekday() == 6:  # Domingo
            slots = []
            fecha_ctx = "Domingo (no atendemos)"
        else:
            slots = await appointments.get_slots_disponibles(fecha_obj)
            fecha_ctx = fecha_obj.strftime("%A %d de %B")

    # Generar respuesta con IA
    ai_response = await generate_response(
        state=state,
        user_message=message,
        slots_disponibles=slots,
        fecha_contexto=fecha_ctx,
    )

    # Verificar si la IA detecto una cita completa
    cita_data = extract_appointment_json(ai_response)
    clean_text = clean_response(ai_response)

    if cita_data:
        # Crear la cita en BD
        try:
            nombre = cita_data.get("nombre", state.contact_name or "")
            telefono = cita_data.get("telefono", state.contact_phone or "")
            fecha = date.fromisoformat(cita_data["fecha"])
            hora_parts = cita_data["hora"].split(":")
            from datetime import time
            hora = time(int(hora_parts[0]), int(hora_parts[1]))

            cita = Cita(
                nombre_paciente=nombre,
                telefono=telefono,
                fecha=fecha,
                hora=hora,
                motivo=cita_data.get("motivo", "Consulta neumología"),
                canal=state.canal,
                conversation_id=conversation_id,
                contact_id=contact_id,
            )
            cita_id = await appointments.crear_cita(cita)
            log.info(f"[Conv {conversation_id}] CITA CREADA #{cita_id}: {nombre} {fecha} {hora}")

            # Etiquetar conversacion en Chatwoot
            await add_label(conversation_id, "cita_agendada")

            # Limpiar estado (conversacion completada)
            await delete_state(conversation_id)
        except Exception as e:
            log.error(f"Error creando cita: {e}")

    # Enviar respuesta al paciente via Chatwoot
    await send_message(conversation_id, clean_text)

    # Guardar respuesta en historial
    if not cita_data:  # Si no se creo cita, mantener estado
        state = await add_message(state, "assistant", clean_text)

    return {"ok": True}


# ─── API Panel de Citas ───

@app.get("/api/citas/hoy")
async def citas_hoy():
    """Citas de hoy para el panel."""
    citas = await appointments.get_citas_dia(date.today())
    stats = await appointments.stats_dia(date.today())
    return {"fecha": date.today().isoformat(), "stats": stats, "citas": citas}


@app.get("/api/citas/{fecha}")
async def citas_por_fecha(fecha: str):
    """Citas de una fecha especifica."""
    try:
        f = date.fromisoformat(fecha)
    except ValueError:
        raise HTTPException(400, "Formato de fecha invalido. Usar YYYY-MM-DD")
    citas = await appointments.get_citas_dia(f)
    stats = await appointments.stats_dia(f)
    return {"fecha": fecha, "stats": stats, "citas": citas}


@app.get("/api/citas/semana/{fecha}")
async def citas_semana(fecha: str):
    """Citas de la semana."""
    try:
        f = date.fromisoformat(fecha)
    except ValueError:
        raise HTTPException(400, "Formato invalido")
    inicio = f - timedelta(days=f.weekday())
    fin = inicio + timedelta(days=6)
    citas = await appointments.get_citas_rango(inicio, fin)
    return {"desde": inicio.isoformat(), "hasta": fin.isoformat(), "citas": citas}


@app.patch("/api/citas/{cita_id}/estado")
async def cambiar_estado(cita_id: int, request: Request):
    """Cambiar estado de una cita (panel del equipo)."""
    body = await request.json()
    estado = body.get("estado")
    notas = body.get("notas")
    try:
        estado_enum = EstadoCita(estado)
    except ValueError:
        raise HTTPException(400, f"Estado invalido. Opciones: {[e.value for e in EstadoCita]}")
    await appointments.actualizar_estado(cita_id, estado_enum, notas)
    return {"ok": True, "cita_id": cita_id, "nuevo_estado": estado}


@app.get("/api/stats/hoy")
async def stats_hoy():
    """Estadisticas del dia."""
    return await appointments.stats_dia(date.today())


# ─── Panel Web ───

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/panel")
async def panel():
    """Panel de citas para el equipo del doctor."""
    return FileResponse(STATIC_DIR / "panel.html")


# ─── Health ───

@app.get("/health")
async def health():
    return {"status": "ok", "service": "doc-c-agent", "clinica": config.CLINICA_NOMBRE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.PORT, reload=True)
