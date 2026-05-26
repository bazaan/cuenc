"""
Agente IA — Clínica Respira Vida (Doc C)
FastAPI: webhook Chatwoot + API panel de citas
"""

import asyncio
import json
import logging
import os
import re
from datetime import date, datetime, timedelta
from contextlib import asynccontextmanager

from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import config
from models import ConversationState, Cita, EstadoCita, Canal
from conversation import get_state, save_state, delete_state, add_message
from ai_engine import generate_response, extract_appointment_json, extract_supervisor_tag, clean_response, get_followup_message
from chatwoot_client import send_message, add_label, assign_team
from transcriber import transcribe_audio, describe_image
import appointments
import auth
import gcal
from config import GOOGLE_CLIENT_ID

# ─── Control: pausar/reanudar IA ───
AI_ENABLED = os.getenv("AI_ENABLED", "false").lower() == "true"  # Controlar vía env o API

# ─── Debounce: acumular mensajes rápidos ───
DEBOUNCE_SECONDS = 30
_pending_tasks: dict[int, asyncio.Task] = {}  # conversation_id -> Task
_pending_messages: dict[int, list[dict]] = {}  # conversation_id -> [payloads]
_processing: set[int] = set()  # conversation_ids currently being processed

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("docc-agent")


# ─── Startup / Shutdown ───

_followup_task: asyncio.Task | None = None

async def _followup_loop():
    """Background: revisa cada 15 min si hay conversaciones que necesitan seguimiento."""
    while True:
        try:
            await asyncio.sleep(900)  # cada 15 minutos

            # Cerrar seguimientos expirados (>24h)
            await appointments.cerrar_seguimientos_expirados()

            # No enviar seguimientos fuera de horario (9pm-8am Lima)
            from zoneinfo import ZoneInfo
            hora_lima = datetime.now(ZoneInfo("America/Lima")).hour
            if hora_lima >= 21 or hora_lima < 8:
                continue

            pendientes = await appointments.get_seguimientos_pendientes()
            for seg in pendientes:
                conv_id = seg["conversation_id"]

                # Safety net: verificar handoff en Redis antes de enviar
                state = await get_state(conv_id)
                if state and (state.handoff or state.cita_creada):
                    log.info(f"[Seguimiento] Conv {conv_id} en handoff/cita_creada — cerrando seguimiento")
                    await appointments.cerrar_seguimiento(conv_id)
                    continue

                num = seg["seguimiento_num"] + 1
                msg = get_followup_message(num)

                log.info(f"[Seguimiento #{num}] Conv {conv_id} ({seg['contact_name']}): {msg[:60]}")
                await send_message(conv_id, msg)
                await appointments.marcar_seguimiento_enviado(seg["id"], num)

                # Registrar en log de ejecuciones
                await appointments.registrar_ejecucion(
                    conversation_id=conv_id,
                    contact_name=seg["contact_name"] or "",
                    canal="whatsapp",
                    mensaje_usuario=f"[SEGUIMIENTO {num} - sin respuesta]",
                    respuesta_agente=msg,
                    tipo="seguimiento",
                    cita_creada=False,
                )
                await asyncio.sleep(2)  # pausa entre envios

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Error en followup loop: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _followup_task
    log.info("Iniciando agente Doc C...")
    await appointments.init_db()
    pool = await appointments.get_pool()
    await auth.init_users_table(pool)
    _followup_task = asyncio.create_task(_followup_loop())
    log.info(f"Agente listo en puerto {config.PORT} (followup loop activo)")
    yield
    if _followup_task:
        _followup_task.cancel()
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


def _hoy_lima() -> date:
    """Fecha actual en timezone Lima."""
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/Lima")).date()


def parse_fecha_from_text(text: str) -> date | None:
    """Intenta extraer una fecha del texto del usuario."""
    hoy = _hoy_lima()
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

async def _process_accumulated_messages(conversation_id: int, payloads: list[dict]):
    """Procesa todos los mensajes acumulados de una conversación como uno solo."""
    # Usar el último payload para datos de contacto (más actualizado)
    last_payload = payloads[-1]
    messages_texts = [p["_message"] for p in payloads]
    combined_message = "\n".join(messages_texts)

    conversation = last_payload.get("conversation", {})
    inbox_id = conversation.get("inbox_id")
    channel_raw = conversation.get("channel", {})
    channel_type = channel_raw.get("type", "") if isinstance(channel_raw, dict) else str(channel_raw)

    sender = last_payload.get("sender", {})
    contact_id = sender.get("id")
    contact_name = sender.get("name", "")
    contact_phone = sender.get("phone_number", "")

    # Fallback: buscar datos en conversation.meta.sender (Agent Bot payload)
    conv_meta = conversation.get("meta", {})
    meta_sender = conv_meta.get("sender", {})
    if not contact_name and meta_sender.get("name"):
        contact_name = meta_sender["name"]
    if not contact_phone and meta_sender.get("phone_number"):
        contact_phone = meta_sender["phone_number"]
    # Fallback: contact_inbox source_id (numero WA)
    if not contact_phone:
        contact_inbox = conversation.get("contact_inbox", {})
        if contact_inbox.get("source_id"):
            contact_phone = contact_inbox["source_id"]

    log.info(f"[Conv {conversation_id}] Procesando {len(payloads)} mensaje(s) acumulados: {combined_message[:120]}")

    # Obtener o crear estado de conversacion
    state = await get_state(conversation_id)

    # Si la conversación está en handoff, la IA no responde
    if state and state.handoff:
        log.info(f"[Conv {conversation_id}] HANDOFF activo — IA no responde (supervisor tiene control)")
        return

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

    # Agregar mensaje combinado al historial
    state = await add_message(state, "user", combined_message)

    # Detectar fecha/hora en TODOS los mensajes
    fecha_detectada = None
    hora_detectada = None
    for txt in messages_texts:
        f = parse_fecha_from_text(txt)
        h = parse_hora_from_text(txt)
        if f:
            fecha_detectada = f
        if h:
            hora_detectada = h

    if fecha_detectada:
        state.fecha_elegida = fecha_detectada.isoformat()
    if hora_detectada:
        state.hora_elegida = hora_detectada

    # Obtener slots disponibles si hay fecha (verifica BD + Google Calendar)
    slots = None
    fecha_ctx = None
    if state.fecha_elegida:
        fecha_obj = date.fromisoformat(state.fecha_elegida)
        if fecha_obj.weekday() == 6:
            slots = []
            fecha_ctx = "Domingo (no atendemos)"
        else:
            gcal_busy = []
            gcal_token = await appointments.get_gcal_token()
            if gcal_token:
                gcal_busy = await gcal.get_busy_slots(gcal_token, fecha_obj)
            slots = await appointments.get_slots_disponibles(fecha_obj, gcal_busy=gcal_busy)
            dias_es = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
            meses_es = ["", "enero", "febrero", "marzo", "abril", "mayo", "junio",
                        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
            fecha_ctx = f"{dias_es[fecha_obj.weekday()]} {fecha_obj.day} de {meses_es[fecha_obj.month]}"

    # Generar respuesta con IA
    ai_response = await generate_response(
        state=state,
        user_message=combined_message,
        slots_disponibles=slots,
        fecha_contexto=fecha_ctx,
    )

    # Verificar si la IA detecto una cita completa
    cita_data = extract_appointment_json(ai_response)
    clean_text = clean_response(ai_response)

    # Prevenir citas duplicadas: si ya se creó una cita en esta conversación, ignorar
    if cita_data and state.cita_creada:
        log.warning(f"[Conv {conversation_id}] CITA DUPLICADA ignorada — ya existe cita en esta conversación")
        cita_data = None

    if cita_data:
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

            # Crear evento en Google Calendar si esta conectado (pre-agenda = pendiente)
            gcal_token = await appointments.get_gcal_token()
            if gcal_token:
                gcal_eid = await gcal.create_event(
                    gcal_token, nombre, telefono, fecha, hora,
                    motivo=cita_data.get("motivo", "Consulta"),
                    estado="pendiente",
                )
                if gcal_eid:
                    await appointments.update_cita_gcal_id(cita_id, gcal_eid)

            await add_label(conversation_id, "cita_agendada")
            # Handoff automático: IA deja de responder, equipo toma control
            state.cita_creada = True
            state.handoff = True
            log.info(f"[Conv {conversation_id}] HANDOFF automático — cita creada, supervisor toma control")
            # Cerrar seguimiento para que no reciba follow-ups
            await appointments.cerrar_seguimiento(conversation_id)
            # Asignar a team en Chatwoot si está configurado
            if config.CHATWOOT_TEAM_ID:
                await assign_team(conversation_id, config.CHATWOOT_TEAM_ID)
        except Exception as e:
            log.error(f"Error creando cita: {e}")

    # Detectar escalamiento a supervisor (IA decidió pasar a humano)
    supervisor_motivo = extract_supervisor_tag(ai_response)
    if supervisor_motivo and not state.handoff:
        state.handoff = True
        log.info(f"[Conv {conversation_id}] HANDOFF por IA — motivo: {supervisor_motivo}")
        await add_label(conversation_id, "supervisor")
        # Cerrar seguimiento para que no reciba follow-ups
        await appointments.cerrar_seguimiento(conversation_id)
        if config.CHATWOOT_TEAM_ID:
            await assign_team(conversation_id, config.CHATWOOT_TEAM_ID)
        # Nota privada para el equipo
        await send_message(conversation_id, f"🔔 IA escaló a supervisor: {supervisor_motivo}", private=True)

    # Enviar respuesta al paciente via Chatwoot
    await send_message(conversation_id, clean_text)

    # Registrar ejecucion en DB
    msg_tipo = "texto"
    for att in payloads[0].get("attachments", []) if payloads else []:
        if att.get("file_type") == "audio":
            msg_tipo = "audio"
        elif att.get("file_type") == "image":
            msg_tipo = "imagen"
    await appointments.registrar_ejecucion(
        conversation_id=conversation_id,
        contact_name=state.contact_name or "",
        canal=state.canal.value if state.canal else "whatsapp",
        mensaje_usuario=combined_message[:500],
        respuesta_agente=clean_text[:500],
        tipo=msg_tipo,
        cita_creada=bool(cita_data),
        contact_phone=state.contact_phone or contact_phone or "",
    )

    # Detectar si paciente mostró interés en agendar (para seguimiento)
    palabras_interes = ["cita", "consulta", "agendar", "reservar", "turno", "horario", "disponib", "mañana", "hoy", "lunes", "martes", "miercoles", "jueves", "viernes", "sabado"]
    msg_lower = combined_message.lower()
    paciente_interesado = any(p in msg_lower for p in palabras_interes) or bool(fecha_detectada)

    # Tracking de seguimiento (resetea timer si usuario responde, cierra si cita creada)
    await appointments.upsert_seguimiento(
        conversation_id=conversation_id,
        contact_name=state.contact_name or contact_name or "",
        cita_creada=bool(cita_data),
        interesado=paciente_interesado,
    )

    # Guardar respuesta en historial (siempre, incluso después de cita)
    state = await add_message(state, "assistant", clean_text)


async def _debounce_fire(conversation_id: int):
    """Espera DEBOUNCE_SECONDS y luego procesa los mensajes acumulados."""
    await asyncio.sleep(DEBOUNCE_SECONDS)
    # Tomar y limpiar mensajes pendientes
    payloads = _pending_messages.pop(conversation_id, [])
    _pending_tasks.pop(conversation_id, None)
    if not payloads:
        return
    # Evitar procesamiento concurrente de la misma conversación
    if conversation_id in _processing:
        log.warning(f"[Conv {conversation_id}] Ya se está procesando, re-encolando {len(payloads)} msgs")
        _pending_messages.setdefault(conversation_id, []).extend(payloads)
        _pending_tasks[conversation_id] = asyncio.create_task(_debounce_fire(conversation_id))
        return
    _processing.add(conversation_id)
    try:
        await _process_accumulated_messages(conversation_id, payloads)
    except Exception as e:
        log.error(f"[Conv {conversation_id}] Error procesando mensajes acumulados: {e}")
    finally:
        _processing.discard(conversation_id)


@app.post("/webhook/chatwoot")
async def webhook_chatwoot(request: Request):
    """Recibe mensajes desde Chatwoot. Acumula mensajes rápidos (debounce 3s)."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"ok": False}, status_code=400)

    event = payload.get("event")
    log.info(f"[Webhook] event={event} message_type={payload.get('message_type')} content={str(payload.get('content', ''))[:80]}")

    if not AI_ENABLED:
        log.info("[Webhook] IA pausada — ignorando mensaje")
        return {"ok": True, "skipped": "ai_disabled"}

    if event != "message_created":
        return {"ok": True, "skipped": "not_message_created"}

    message = payload.get("content", "")
    message_type = payload.get("message_type")

    if message_type != "incoming":
        return {"ok": True, "skipped": "not_incoming"}

    # Detectar attachments (audio / imagen)
    attachments = payload.get("attachments", [])
    if attachments:
        log.info(f"[Webhook] Attachments raw: {attachments}")
    extra_parts = []
    for att in attachments:
        file_type = att.get("file_type", "")
        data_url = att.get("data_url", "")
        if not data_url:
            continue

        if file_type == "audio":
            log.info(f"[Webhook] Audio detectado, URL: {data_url[:120]}")
            transcription = await transcribe_audio(data_url)
            if transcription:
                message = transcription
            else:
                log.warning("[Webhook] No se pudo transcribir el audio")
                return {"ok": True, "skipped": "audio_transcription_failed"}

        elif file_type == "image":
            log.info(f"[Webhook] Imagen detectada, analizando...")
            description = await describe_image(data_url)
            if description:
                extra_parts.append(f"[Imagen enviada: {description}]")

    if extra_parts:
        message = (message or "").strip()
        message = f"{message}\n{chr(10).join(extra_parts)}".strip() if message else "\n".join(extra_parts)

    if not message or not message.strip():
        return {"ok": True, "skipped": "empty_message"}

    conversation = payload.get("conversation", {})
    conversation_id = conversation.get("id")
    sender = payload.get("sender", {})
    contact_id = sender.get("id")

    # Agent Bot: conversation_id puede venir en conversation.messages[0].conversation_id
    if not conversation_id:
        msgs = conversation.get("messages", [])
        if msgs:
            conversation_id = msgs[0].get("conversation_id")

    # Agent Bot: contact_id puede venir en conversation.meta.sender.id
    if not contact_id:
        meta_sender = conversation.get("meta", {}).get("sender", {})
        contact_id = meta_sender.get("id")

    if not conversation_id:
        return {"ok": True, "skipped": "no_conversation_id"}

    log.info(f"[Conv {conversation_id}] Mensaje recibido (debounce): {message[:80]}")

    # Guardar texto limpio en el payload para uso interno
    payload["_message"] = message.strip()

    # Acumular mensaje
    if conversation_id not in _pending_messages:
        _pending_messages[conversation_id] = []
    _pending_messages[conversation_id].append(payload)

    # Cancelar timer anterior si existe
    existing_task = _pending_tasks.get(conversation_id)
    if existing_task and not existing_task.done():
        existing_task.cancel()
        log.info(f"[Conv {conversation_id}] Timer reiniciado ({len(_pending_messages[conversation_id])} msgs acumulados)")

    # Iniciar nuevo timer de 3 segundos
    _pending_tasks[conversation_id] = asyncio.create_task(_debounce_fire(conversation_id))

    return {"ok": True}


# ─── API Panel de Citas ───

@app.get("/api/citas/hoy")
async def citas_hoy():
    """Citas de hoy para el panel."""
    citas = await appointments.get_citas_dia(_hoy_lima())
    stats = await appointments.stats_dia(_hoy_lima())
    return {"fecha": _hoy_lima().isoformat(), "stats": stats, "citas": citas}


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


@app.post("/api/citas")
async def crear_cita_manual(request: Request):
    """Crear cita manualmente desde el dashboard."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.decode_token(token)
    if not payload:
        raise HTTPException(401, "Token invalido")
    body = await request.json()
    try:
        from datetime import time as dt_time
        fecha = date.fromisoformat(body["fecha"])
        hora_parts = body["hora"].split(":")
        hora = dt_time(int(hora_parts[0]), int(hora_parts[1]))
        cita = Cita(
            nombre_paciente=body["nombre"],
            telefono=body.get("telefono", ""),
            fecha=fecha,
            hora=hora,
            motivo=body.get("motivo", ""),
            canal=Canal.WEB,
            estado=EstadoCita(body.get("estado", "pendiente")),
            notas_equipo=body.get("notas", ""),
        )
        cita_id = await appointments.crear_cita(cita)

        # Crear evento en Google Calendar si conectado
        gcal_token = await appointments.get_gcal_token()
        if gcal_token:
            gcal_eid = await gcal.create_event(
                gcal_token, body["nombre"], body.get("telefono", ""),
                fecha, hora, motivo=body.get("motivo", ""),
                estado=body.get("estado", "pendiente"),
            )
            if gcal_eid:
                await appointments.update_cita_gcal_id(cita_id, gcal_eid)

        return {"ok": True, "id": cita_id}
    except KeyError as e:
        raise HTTPException(400, f"Campo requerido faltante: {e}")
    except ValueError as e:
        raise HTTPException(400, f"Valor invalido: {e}")


@app.put("/api/citas/{cita_id}")
async def editar_cita(cita_id: int, request: Request):
    """Editar una cita existente desde el dashboard."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.decode_token(token)
    if not payload:
        raise HTTPException(401, "Token invalido")
    body = await request.json()
    pool = await appointments.get_pool()
    async with pool.acquire() as conn:
        sets = []
        vals = []
        idx = 1
        for field in ("nombre_paciente", "telefono", "motivo", "notas_equipo"):
            if field in body:
                sets.append(f"{field} = ${idx}")
                vals.append(body[field])
                idx += 1
        if "fecha" in body:
            sets.append(f"fecha = ${idx}")
            vals.append(date.fromisoformat(body["fecha"]))
            idx += 1
        if "hora" in body:
            from datetime import time as dt_time
            hp = body["hora"].split(":")
            sets.append(f"hora = ${idx}")
            vals.append(dt_time(int(hp[0]), int(hp[1])))
            idx += 1
        if "estado" in body:
            sets.append(f"estado = ${idx}")
            vals.append(body["estado"])
            idx += 1
        if not sets:
            raise HTTPException(400, "Nada que actualizar")
        vals.append(cita_id)
        query = f"UPDATE citas SET {', '.join(sets)} WHERE id = ${idx}"
        await conn.execute(query, *vals)

    # Sincronizar cambios con Google Calendar
    gcal_token = await appointments.get_gcal_token()
    if gcal_token:
        cita = await appointments.get_cita(cita_id)
        if cita and cita.get("gcal_event_id"):
            from datetime import time as dt_time
            gcal_updates = {}
            if "nombre_paciente" in body:
                gcal_updates["nombre"] = body["nombre_paciente"]
            if "telefono" in body:
                gcal_updates["telefono"] = body["telefono"]
            if "motivo" in body:
                gcal_updates["motivo"] = body["motivo"]
            if "estado" in body:
                if body["estado"] == "cancelada":
                    await gcal.delete_event(gcal_token, cita["gcal_event_id"])
                else:
                    gcal_updates["estado"] = body["estado"]
            if "fecha" in body and "hora" in body:
                hp = body["hora"].split(":")
                gcal_updates["fecha"] = date.fromisoformat(body["fecha"])
                gcal_updates["hora"] = dt_time(int(hp[0]), int(hp[1]))
            elif "fecha" in body:
                gcal_updates["fecha"] = date.fromisoformat(body["fecha"])
                gcal_updates["hora"] = cita["hora"]
            elif "hora" in body:
                hp = body["hora"].split(":")
                gcal_updates["fecha"] = cita["fecha"]
                gcal_updates["hora"] = dt_time(int(hp[0]), int(hp[1]))
            if gcal_updates and body.get("estado") != "cancelada":
                await gcal.update_event(gcal_token, cita["gcal_event_id"], **gcal_updates)
        elif cita and not cita.get("gcal_event_id") and body.get("estado") != "cancelada":
            # Cita sin GCal event → crear uno
            eid = await gcal.create_event(
                gcal_token, cita["nombre_paciente"], cita.get("telefono", ""),
                cita["fecha"], cita["hora"],
                motivo=cita.get("motivo", ""), estado=cita.get("estado", "pendiente"),
            )
            if eid:
                await appointments.update_cita_gcal_id(cita_id, eid)

    return {"ok": True, "cita_id": cita_id}


@app.patch("/api/citas/{cita_id}/estado")
async def cambiar_estado(cita_id: int, request: Request):
    """Cambiar estado de una cita (panel del equipo). Sincroniza con GCal."""
    body = await request.json()
    estado = body.get("estado")
    notas = body.get("notas")
    try:
        estado_enum = EstadoCita(estado)
    except ValueError:
        raise HTTPException(400, f"Estado invalido. Opciones: {[e.value for e in EstadoCita]}")
    await appointments.actualizar_estado(cita_id, estado_enum, notas)

    # Sincronizar con Google Calendar
    gcal_token = await appointments.get_gcal_token()
    if gcal_token:
        cita = await appointments.get_cita(cita_id)
        if cita:
            gcal_eid = cita.get("gcal_event_id")
            if estado == "cancelada" and gcal_eid:
                # Cancelada → eliminar de GCal
                await gcal.delete_event(gcal_token, gcal_eid)
            elif gcal_eid:
                # Actualizar color en GCal segun nuevo estado
                await gcal.update_event(gcal_token, gcal_eid, estado=estado)
            elif estado != "cancelada":
                # No tenia evento GCal → crear uno (pre-gcal citas)
                eid = await gcal.create_event(
                    gcal_token, cita["nombre_paciente"], cita.get("telefono", ""),
                    cita["fecha"], cita["hora"],
                    motivo=cita.get("motivo", ""), estado=estado,
                )
                if eid:
                    await appointments.update_cita_gcal_id(cita_id, eid)

    return {"ok": True, "cita_id": cita_id, "nuevo_estado": estado}


@app.get("/api/stats/hoy")
async def stats_hoy():
    """Estadisticas del dia."""
    return await appointments.stats_dia(_hoy_lima())


@app.get("/api/ejecuciones")
async def get_ejecuciones(limit: int = 50):
    """Log de ejecuciones del agente."""
    rows = await appointments.get_ejecuciones(limit)
    result = []
    for r in rows:
        item = dict(r)
        if item.get("created_at"):
            item["created_at"] = item["created_at"].isoformat()
        result.append(item)
    return {"ejecuciones": result}


@app.get("/api/conversaciones")
async def get_conversaciones(limit: int = 50):
    """Lista de conversaciones agrupadas."""
    rows = await appointments.get_conversaciones(limit)
    result = []
    for r in rows:
        item = dict(r)
        for k in ("primera_interaccion", "ultima_interaccion"):
            if item.get(k):
                item[k] = item[k].isoformat()
        result.append(item)
    return {"conversaciones": result}


@app.get("/api/conversaciones/{conversation_id}")
async def get_hilo(conversation_id: int):
    """Hilo completo de una conversacion."""
    rows = await appointments.get_hilo_conversacion(conversation_id)
    result = []
    for r in rows:
        item = dict(r)
        if item.get("created_at"):
            item["created_at"] = item["created_at"].isoformat()
        result.append(item)
    return {"conversation_id": conversation_id, "mensajes": result}


# ─── Auth API ───

@app.post("/api/auth/login")
async def login(request: Request):
    body = await request.json()
    username = body.get("username", "")
    password = body.get("password", "")
    pool = await appointments.get_pool()
    user = await auth.authenticate(pool, username, password)
    if not user:
        raise HTTPException(401, "Credenciales invalidas")
    token = auth.create_token(user["id"], user["username"], user["role"])
    return {"token": token, "user": {"id": user["id"], "username": user["username"], "nombre": user["nombre"], "role": user["role"]}}


@app.get("/api/auth/me")
async def auth_me(request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.decode_token(token)
    if not payload:
        raise HTTPException(401, "Token invalido")
    return {"user": payload}


@app.get("/api/ai/status")
async def ai_status():
    return {"enabled": AI_ENABLED}


@app.post("/api/ai/toggle")
async def ai_toggle(request: Request):
    global AI_ENABLED
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.decode_token(token)
    if not payload:
        raise HTTPException(401, "No autorizado")
    body = await request.json()
    AI_ENABLED = bool(body.get("enabled", False))
    log.info(f"[AI Toggle] IA {'ACTIVADA' if AI_ENABLED else 'PAUSADA'} por {payload.get('username')}")
    return {"ok": True, "enabled": AI_ENABLED}


# ─── Días bloqueados API ───

@app.get("/api/dias-bloqueados")
async def get_dias_bloqueados():
    """Lista días bloqueados."""
    dias = await appointments.get_dias_bloqueados()
    return {"dias": dias}


@app.post("/api/dias-bloqueados")
async def bloquear_dia(request: Request):
    """Bloquear un día (no ofrece slots)."""
    body = await request.json()
    fecha = date.fromisoformat(body["fecha"])
    motivo = body.get("motivo", "lleno")
    await appointments.bloquear_dia(fecha, motivo)
    return {"ok": True, "fecha": body["fecha"], "motivo": motivo}


@app.delete("/api/dias-bloqueados/{fecha}")
async def desbloquear_dia(fecha: str):
    """Desbloquear un día."""
    f = date.fromisoformat(fecha)
    await appointments.desbloquear_dia(f)
    return {"ok": True, "fecha": fecha}


# ─── Handoff API ───

@app.post("/api/conversations/{conversation_id}/handoff")
async def handoff_conversation(conversation_id: int, request: Request):
    """Activa handoff: la IA deja de responder y el supervisor toma control."""
    body = await request.json() if await request.body() else {}
    activate = body.get("activate", True)  # True = handoff ON, False = devolver a IA

    state = await get_state(conversation_id)
    if state is None:
        # Crear estado mínimo para marcar handoff
        state = ConversationState(
            contact_id=0,
            conversation_id=conversation_id,
            handoff=activate,
        )
    else:
        state.handoff = activate

    await save_state(state)

    action = "ACTIVADO" if activate else "DESACTIVADO"
    log.info(f"[Conv {conversation_id}] Handoff {action} manualmente")

    # Cerrar seguimiento si se activa handoff
    if activate:
        await appointments.cerrar_seguimiento(conversation_id)

    # Si se activa y hay team configurado, asignar en Chatwoot
    if activate and config.CHATWOOT_TEAM_ID:
        await assign_team(conversation_id, config.CHATWOOT_TEAM_ID)

    return {"ok": True, "conversation_id": conversation_id, "handoff": activate}


@app.get("/api/conversations/{conversation_id}/handoff")
async def handoff_status(conversation_id: int):
    """Consulta si una conversación está en handoff."""
    state = await get_state(conversation_id)
    is_handoff = state.handoff if state else False
    return {"conversation_id": conversation_id, "handoff": is_handoff}


@app.get("/api/usuarios")
async def get_usuarios(request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.decode_token(token)
    if not payload or payload.get("role") != "admin":
        raise HTTPException(403, "Solo admin")
    pool = await appointments.get_pool()
    users = await auth.get_users(pool)
    for u in users:
        if u.get("created_at"):
            u["created_at"] = u["created_at"].isoformat()
    return {"usuarios": users}


@app.post("/api/usuarios")
async def crear_usuario(request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.decode_token(token)
    if not payload or payload.get("role") != "admin":
        raise HTTPException(403, "Solo admin")
    body = await request.json()
    pool = await appointments.get_pool()
    try:
        uid = await auth.create_user(pool, body["username"], body["password"], body["nombre"], body.get("role", "staff"))
    except Exception:
        raise HTTPException(400, "Username ya existe")
    return {"ok": True, "id": uid}


@app.delete("/api/usuarios/{user_id}")
async def eliminar_usuario(user_id: int, request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.decode_token(token)
    if not payload or payload.get("role") != "admin":
        raise HTTPException(403, "Solo admin")
    pool = await appointments.get_pool()
    await auth.delete_user(pool, user_id)
    return {"ok": True}


# ─── Google Calendar API ───

@app.get("/api/gcal/status")
async def gcal_status(request: Request):
    """Estado de conexion con Google Calendar."""
    token_data = await appointments.get_gcal_token()
    if token_data:
        return {"connected": True, "email": token_data.get("_email", "")}
    return {"connected": False, "email": ""}


@app.get("/api/auth/google")
async def auth_google():
    """Redirige a Google OAuth — login + calendario en un paso."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(400, "Google no configurado")
    url = await gcal.get_auth_url(state="login")
    return RedirectResponse(url=url)


@app.get("/api/gcal/auth")
async def gcal_auth(request: Request):
    """Genera URL de autorizacion de Google (para vincular desde dashboard)."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.decode_token(token)
    if not payload:
        raise HTTPException(401, "Token invalido")
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(400, "Google Calendar no configurado. Falta GOOGLE_CLIENT_ID.")
    url = await gcal.get_auth_url(state="connect:" + str(payload.get("sub", "")))
    return {"auth_url": url}


@app.get("/api/gcal/callback")
async def gcal_callback(request: Request):
    """Callback OAuth2 de Google — login automatico + vincula calendario."""
    code = request.query_params.get("code")
    error = request.query_params.get("error")
    state = request.query_params.get("state", "login")

    if error:
        return RedirectResponse(url="/?error=google_denied")
    if not code:
        raise HTTPException(400, "No authorization code")

    try:
        # Intercambiar code por tokens
        token_data = await gcal.exchange_code(code)

        # Obtener info del usuario Google
        user_info = await gcal.get_user_info(token_data)
        email = user_info.get("email", "")
        name = user_info.get("name", "")

        if not email:
            email = await gcal.get_calendar_email(token_data)

        # Guardar token de Google Calendar (global para la clinica)
        await appointments.save_gcal_token(token_data, email)
        log.info(f"Google Calendar conectado: {email} ({name})")

        if state == "login" or not state.startswith("connect:"):
            # Login flow: crear/encontrar usuario y generar JWT
            pool = await appointments.get_pool()
            db_user = await auth.find_or_create_google_user(pool, email, name)
            jwt_token = auth.create_token(db_user["id"], db_user["username"], db_user["role"])

            # Redirect al dashboard con token (el JS lo guarda en localStorage)
            import urllib.parse
            user_json = urllib.parse.quote(json.dumps({
                "id": db_user["id"],
                "username": db_user["username"],
                "nombre": db_user["nombre"],
                "role": db_user["role"],
            }))
            return RedirectResponse(
                url=f"/dashboard?auth_token={jwt_token}&auth_user={user_json}"
            )
        else:
            # Connect flow: solo vincular calendario, ya esta logueado
            return RedirectResponse(url="/dashboard?gcal=connected")

    except Exception as e:
        log.error(f"GCal callback error: {e}")
        return RedirectResponse(url="/?error=google_error")


@app.get("/api/gcal/events")
async def gcal_events(desde: str, hasta: str):
    """Lista eventos de Google Calendar en un rango."""
    token_data = await appointments.get_gcal_token()
    if not token_data:
        return {"events": [], "connected": False}
    try:
        events = await gcal.list_events(
            token_data, date.fromisoformat(desde), date.fromisoformat(hasta),
        )
        return {"events": events, "connected": True}
    except Exception as e:
        log.error(f"Error fetching gcal events: {e}")
        return {"events": [], "connected": True, "error": str(e)}


@app.delete("/api/gcal/disconnect")
async def gcal_disconnect(request: Request):
    """Desconectar Google Calendar."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.decode_token(token)
    if not payload or payload.get("role") != "admin":
        raise HTTPException(403, "Solo admin")
    await appointments.delete_gcal_token()
    log.info("Google Calendar desconectado")
    return {"ok": True}


# ─── Panel Web ───

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def root():
    """Login page."""
    return FileResponse(STATIC_DIR / "login.html")


@app.get("/dashboard")
async def dashboard_page():
    """Dashboard principal (protegido por JS)."""
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/panel")
async def panel():
    """Panel legacy."""
    return FileResponse(STATIC_DIR / "panel.html")


# ─── Health ───

@app.get("/health")
async def health():
    return {"status": "ok", "service": "doc-c-agent", "clinica": config.CLINICA_NOMBRE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.PORT, reload=True)
