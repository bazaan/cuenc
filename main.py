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
import conversation
from conversation import get_state, save_state, delete_state, add_message
from ai_engine import generate_response, extract_appointment_json, extract_supervisor_tag, clean_response, get_followup_message
from chatwoot_client import send_message, add_label, remove_label, get_conversation_labels, assign_team, set_custom_attributes, get_custom_attributes, ensure_account_labels, list_open_conversations, get_conversation_messages
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

                # Safety net: verificar handoff en Redis + Chatwoot antes de enviar
                state = await get_state(conv_id)
                cw_attrs = await get_custom_attributes(conv_id)
                is_handoff = (state and (state.handoff or state.cita_creada)) or cw_attrs.get("pasar_supervisor") == "si"
                if is_handoff:
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


_watchdog_task: asyncio.Task | None = None
WATCHDOG_INTERVAL = 180  # cada 3 minutos
WATCHDOG_MIN_AGE = 180   # solo mensajes sin respuesta hace >= 3 min
WATCHDOG_MAX_AGE = 3600  # ignorar mensajes de más de 1 hora (probablemente legítimos)


async def _watchdog_loop():
    """Background: detecta conversaciones abiertas con mensajes incoming sin respuesta."""
    while True:
        try:
            await asyncio.sleep(WATCHDOG_INTERVAL)

            if not AI_ENABLED:
                continue

            from zoneinfo import ZoneInfo
            now_lima = datetime.now(ZoneInfo("America/Lima"))
            if now_lima.hour >= 21 or now_lima.hour < 8:
                continue

            # Obtener conversaciones abiertas de Chatwoot
            conversations = await list_open_conversations(page=1)
            # Página 2 si hay muchas
            if len(conversations) >= 25:
                conversations += await list_open_conversations(page=2)

            recovered = 0
            for conv in conversations:
                try:
                    conv_id = conv.get("id")
                    if not conv_id:
                        continue

                    # FILTRO MEMORIA: si ya está en debounce o procesándose, skip
                    if conv_id in _pending_tasks or conv_id in _pending_messages or conv_id in _processing:
                        continue

                    # Obtener últimos mensajes de la conversación
                    msgs = await get_conversation_messages(conv_id, limit=5)
                    if not msgs:
                        continue

                    # Encontrar el último mensaje no-activity
                    last_real = None
                    for m in msgs:
                        if m.get("message_type") in (0, 1):  # 0=incoming, 1=outgoing
                            last_real = m
                            break  # msgs vienen desc, el primero es el más reciente

                    if not last_real or last_real.get("message_type") != 0:
                        continue  # último mensaje es outgoing o no hay mensajes → ok

                    # Verificar antigüedad (>= 3 min sin respuesta)
                    msg_time = last_real.get("created_at")
                    if not msg_time:
                        continue
                    try:
                        if isinstance(msg_time, (int, float)):
                            msg_dt = datetime.fromtimestamp(msg_time, tz=ZoneInfo("UTC"))
                        else:
                            msg_dt = datetime.fromisoformat(str(msg_time).replace("Z", "+00:00"))
                        age_seconds = (datetime.now(ZoneInfo("UTC")) - msg_dt).total_seconds()
                    except (ValueError, TypeError):
                        continue

                    if age_seconds < WATCHDOG_MIN_AGE or age_seconds > WATCHDOG_MAX_AGE:
                        continue

                    # FILTRO HANDOFF: verificar Redis + Chatwoot
                    state = await get_state(conv_id)
                    if state and (state.handoff or state.cita_creada):
                        continue
                    cw_attrs = conv.get("custom_attributes", {})
                    if cw_attrs.get("pasar_supervisor") == "si" or cw_attrs.get("ai_status") in ("supervisor", "cita_agendada"):
                        continue
                    labels = [l.get("title", "") if isinstance(l, dict) else str(l) for l in conv.get("labels", [])]
                    if "supervisor" in labels or "pasar_supervisor" in labels:
                        continue

                    # FILTRO TEAM PHONES
                    sender = last_real.get("sender", {})
                    sender_phone = sender.get("phone_number", "")
                    if sender_phone:
                        sender_digits = ''.join(c for c in sender_phone if c.isdigit())[-9:]
                        team_digits = {''.join(c for c in p if c.isdigit())[-9:] for p in config.TEAM_PHONES}
                        if sender_digits in team_digits:
                            continue

                    # FILTRO BD: verificar si ya respondimos recientemente
                    ultima = await appointments.get_ultima_ejecucion(conv_id)
                    if ultima and ultima.get("created_at"):
                        ult_dt = ultima["created_at"]
                        if hasattr(ult_dt, 'timestamp'):
                            ult_age = (datetime.now(ZoneInfo("UTC")) - ult_dt.astimezone(ZoneInfo("UTC"))).total_seconds()
                            if ult_age < 300:  # respondimos hace < 5 min
                                continue

                    # LOCK REDIS para evitar procesamiento concurrente
                    r = await conversation.get_redis()
                    lock_key = f"docc:watchdog:lock:{conv_id}"
                    locked = await r.set(lock_key, "1", ex=120, nx=True)
                    if not locked:
                        continue

                    # RECUPERAR: extraer mensajes incoming sin respuesta
                    incoming_texts = []
                    for m in msgs:
                        if m.get("message_type") == 0:  # incoming
                            content = m.get("content", "")
                            if content:
                                incoming_texts.append(content)
                        elif m.get("message_type") == 1:  # outgoing → ya hubo respuesta antes
                            break

                    if not incoming_texts:
                        continue

                    incoming_texts.reverse()  # poner en orden cronológico
                    combined = "\n".join(incoming_texts)

                    # Construir payload mínimo compatible con _process_accumulated_messages
                    meta = conv.get("meta", {})
                    meta_sender = meta.get("sender", {})
                    fake_payload = {
                        "_message": combined,
                        "conversation": conv,
                        "sender": meta_sender,
                    }

                    log.info(f"[Watchdog] RECOVERY Conv {conv_id} — {len(incoming_texts)} msg(s) sin respuesta ({int(age_seconds)}s): {combined[:80]}")

                    # Procesar como si fuera un mensaje normal
                    await _process_accumulated_messages(conv_id, [fake_payload])

                    # Registrar alerta
                    await appointments.registrar_alerta(
                        tipo="watchdog_recovery",
                        conversation_id=conv_id,
                        contact_name=meta_sender.get("name", ""),
                        contact_phone=meta_sender.get("phone_number", ""),
                        detalle=f"Recuperados {len(incoming_texts)} mensaje(s) sin respuesta ({int(age_seconds)}s)",
                    )
                    recovered += 1
                    await asyncio.sleep(1)  # pausa entre recuperaciones

                except Exception as e:
                    log.error(f"[Watchdog] Error procesando conv {conv.get('id')}: {e}")

            if recovered:
                log.info(f"[Watchdog] Ciclo completado: {recovered} conversación(es) recuperada(s)")

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"[Watchdog] Error en loop: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _followup_task, _watchdog_task
    log.info("Iniciando agente Doc C...")
    await appointments.init_db()
    pool = await appointments.get_pool()
    await auth.init_users_table(pool)
    _followup_task = asyncio.create_task(_followup_loop())
    _watchdog_task = asyncio.create_task(_watchdog_loop())
    # Crear labels en Chatwoot para uso desde app movil
    await ensure_account_labels([
        {"title": "pasar_supervisor", "description": "Pausar IA y pasar a supervisor", "color": "#E74C3C", "show_on_sidebar": True},
        {"title": "devolver_ia", "description": "Devolver conversacion a la IA", "color": "#27AE60", "show_on_sidebar": True},
        {"title": "supervisor", "description": "Conversacion atendida por supervisor", "color": "#E67E22", "show_on_sidebar": True},
        {"title": "cita_agendada", "description": "Cita agendada por la IA", "color": "#3498DB", "show_on_sidebar": True},
        {"title": "ia_activa", "description": "IA respondiendo activamente", "color": "#2ECC71", "show_on_sidebar": True},
    ])
    log.info(f"Agente listo en puerto {config.PORT} (followup + watchdog loops activos)")
    yield
    if _followup_task:
        _followup_task.cancel()
    if _watchdog_task:
        _watchdog_task.cancel()
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

    meses = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
        "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
    }

    # "3 de junio", "3 junio", "15 de mayo"  (prioridad sobre nombre de día)
    for mes_nombre, mes_num in meses.items():
        match = re.search(rf"(\d{{1,2}})\s*(?:de\s+)?{mes_nombre}", text_lower)
        if match:
            try:
                d = int(match.group(1))
                resultado = date(hoy.year, mes_num, d)
                if resultado < hoy:
                    resultado = date(hoy.year + 1, mes_num, d)
                return resultado
            except ValueError:
                pass

    # Formato DD/MM o DD-MM
    match = re.search(r"(\d{1,2})[/-](\d{1,2})", text)
    if match:
        try:
            d, m = int(match.group(1)), int(match.group(2))
            resultado = date(hoy.year, m, d)
            if resultado < hoy:
                resultado = date(hoy.year + 1, m, d)
            return resultado
        except ValueError:
            pass

    # Dias de la semana (solo si no se detectó fecha específica arriba)
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

    # Si la conversación está en handoff, verificar si expiró (auto-release 2h)
    HANDOFF_TIMEOUT_HOURS = 2
    if state and state.handoff:
        handoff_expired = False
        if state.handoff_at:
            try:
                handoff_time = datetime.fromisoformat(state.handoff_at)
                elapsed = datetime.now(handoff_time.tzinfo if handoff_time.tzinfo else None) - handoff_time
                if elapsed.total_seconds() > HANDOFF_TIMEOUT_HOURS * 3600:
                    handoff_expired = True
            except (ValueError, TypeError):
                handoff_expired = True  # timestamp corrupto, liberar
        else:
            handoff_expired = True  # sin timestamp = handoff viejo, liberar

        if handoff_expired:
            state.handoff = False
            state.handoff_at = None
            state.cita_creada = False
            await save_state(state)
            await set_custom_attributes(conversation_id, {"ai_status": "ia_activa", "pasar_supervisor": "no"})
            await remove_label(conversation_id, "supervisor")
            await remove_label(conversation_id, "pasar_supervisor")
            log.info(f"[Conv {conversation_id}] HANDOFF expirado (>{HANDOFF_TIMEOUT_HOURS}h) — IA retoma control automaticamente")
        else:
            log.info(f"[Conv {conversation_id}] HANDOFF activo (Redis, desde {state.handoff_at}) — IA no responde")
            return

    # CHECK Chatwoot: verificar si el equipo activó "pasar_supervisor" desde Chatwoot
    cw_attrs = await get_custom_attributes(conversation_id)
    if cw_attrs.get("pasar_supervisor") == "si":
        # Verificar si tambien expiró en Chatwoot
        if state and state.handoff_at:
            try:
                handoff_time = datetime.fromisoformat(state.handoff_at)
                elapsed = datetime.now(handoff_time.tzinfo if handoff_time.tzinfo else None) - handoff_time
                if elapsed.total_seconds() > HANDOFF_TIMEOUT_HOURS * 3600:
                    await set_custom_attributes(conversation_id, {"ai_status": "ia_activa", "pasar_supervisor": "no"})
                    await remove_label(conversation_id, "supervisor")
                    await remove_label(conversation_id, "pasar_supervisor")
                    log.info(f"[Conv {conversation_id}] HANDOFF Chatwoot expirado — IA retoma control")
                    # Continuar procesando el mensaje
                else:
                    log.info(f"[Conv {conversation_id}] HANDOFF detectado en Chatwoot (pasar_supervisor=si) — IA no responde")
                    return
            except (ValueError, TypeError):
                pass  # timestamp corrupto, dejar pasar
        else:
            # Sin timestamp de handoff — es un handoff nuevo, respetar
            if state is None:
                state = ConversationState(contact_id=contact_id or 0, conversation_id=conversation_id)
            state.handoff = True
            state.handoff_at = datetime.now().isoformat()
            await save_state(state)
            log.info(f"[Conv {conversation_id}] HANDOFF nuevo detectado en Chatwoot (pasar_supervisor=si) — IA no responde")
            return

    is_first_message = state is None
    if state is None:
        state = ConversationState(
            contact_id=contact_id,
            contact_name=contact_name or None,
            contact_phone=contact_phone or None,
            conversation_id=conversation_id,
            inbox_id=inbox_id,
            canal=detect_canal(channel_type),
        )
        # Marcar en Chatwoot que la IA está activa en esta conversación
        await set_custom_attributes(conversation_id, {"ai_status": "ia_activa", "pasar_supervisor": "no"})

    # Enviar saludo inicial SIEMPRE que sea primer mensaje (nueva o reciclada)
    # El paciente merece un saludo cordial cada vez que inicia contacto
    if is_first_message:
        saludo = (
            "Gracias por comunicarte con nosotros.\n"
            "Somos la *Clínica Respira Vida* especializada en Neumología y Alergias Respiratorias.\n\n"
            "Por favor escriba la opción que necesite:\n\n"
            "▪️ Costos y disponibilidad de citas\n"
            "▪️ Horario de citas\n"
            "▪️ Dirección\n"
            "▪️ Interconsulta laboral\n"
            "▪️ Otros\n\n"
            "Recuerde que la atención es solo presencial y con previa cita."
        )
        await send_message(conversation_id, saludo)
        state = await add_message(state, "assistant", saludo)
        # Guardar el mensaje del usuario en historial antes de retornar
        if contact_name and not state.contact_name:
            state.contact_name = contact_name
        if contact_phone and not state.contact_phone:
            state.contact_phone = contact_phone
        state = await add_message(state, "user", combined_message)
        await save_state(state)
        # Registrar en ejecuciones para que el watchdog no lo recoja
        await appointments.registrar_ejecucion(
            conversation_id=conversation_id,
            contact_name=contact_name or "",
            canal=state.canal.value if state.canal else "whatsapp",
            mensaje_usuario=combined_message[:500],
            respuesta_agente=saludo[:500],
            tipo="saludo",
            cita_creada=False,
            contact_phone=contact_phone or "",
        )
        log.info(f"[Conv {conversation_id}] Saludo inicial enviado — esperando respuesta del paciente")
        return

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

    # Obtener slots disponibles — SIEMPRE pasar contexto de disponibilidad
    slots = None
    fecha_ctx = None
    dias_es = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    meses_es = ["", "enero", "febrero", "marzo", "abril", "mayo", "junio",
                "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

    async def _get_slots_for_date(fecha_obj):
        if fecha_obj.weekday() == 6:
            return [], "Domingo (no atendemos)"
        gcal_busy = None
        gcal_token = await appointments.get_gcal_token()
        if gcal_token:
            gcal_busy = await gcal.get_busy_slots(gcal_token, fecha_obj)
        # Auto-detectar y bloquear turnos llenos (GCal + DB)
        await appointments.detectar_y_bloquear_turno_lleno(fecha_obj, gcal_busy)
        s = await appointments.get_slots_disponibles(fecha_obj, gcal_busy=gcal_busy)
        ctx = f"{dias_es[fecha_obj.weekday()]} {fecha_obj.day} de {meses_es[fecha_obj.month]}"
        return s, ctx

    from config import CITAS_DIA_LLENO

    if state.fecha_elegida:
        # Paciente pidió fecha específica — mostrar slots de ese día
        fecha_obj = date.fromisoformat(state.fecha_elegida)
        slots, fecha_ctx = await _get_slots_for_date(fecha_obj)
        # Si no hay slots para la fecha pedida, buscar siguiente día con disponibilidad
        if not slots:
            hoy_fecha = _hoy_lima()
            alternativa_info = []
            alternativa_info.append(f"{fecha_ctx}: NO hay horarios disponibles")
            cursor = fecha_obj + timedelta(days=1)
            dias_buscados = 0
            while dias_buscados < 7:
                if cursor.weekday() != 6:
                    s_alt, ctx_alt = await _get_slots_for_date(cursor)
                    if s_alt:
                        if cursor == hoy_fecha:
                            lbl = f"Hoy ({ctx_alt})"
                        elif cursor == hoy_fecha + timedelta(days=1):
                            lbl = f"Mañana ({ctx_alt})"
                        else:
                            lbl = ctx_alt.capitalize()
                        alternativa_info.append(f"Siguiente disponible — {lbl}: {', '.join(s_alt)}")
                        break
                cursor += timedelta(days=1)
                dias_buscados += 1
            fecha_ctx = "fecha solicitada y alternativa"
            slots = alternativa_info
    else:
        # Sin fecha detectada: PRIORIZAR HOY → MAÑANA → después
        hoy_fecha = _hoy_lima()
        slots_info = []
        dias_revisados = 0
        dias_ofrecidos = 0
        fecha_cursor = hoy_fecha
        while dias_ofrecidos < 2 and dias_revisados < 7:
            if fecha_cursor.weekday() != 6:  # Saltar domingos
                n_citas = await appointments.contar_citas_dia(fecha_cursor)
                s, ctx = await _get_slots_for_date(fecha_cursor)
                if fecha_cursor == hoy_fecha:
                    label = f"⭐ Hoy ({ctx}) — PRIORIDAD, ofrecer primero"
                elif fecha_cursor == hoy_fecha + timedelta(days=1):
                    label = f"⭐ Mañana ({ctx}) — PRIORIDAD, ofrecer si hoy está lleno"
                else:
                    label = f"{ctx.capitalize()} — solo ofrecer si hoy y mañana están llenos"
                if s and n_citas < CITAS_DIA_LLENO:
                    slots_info.append(f"{label}: {', '.join(s)}")
                    dias_ofrecidos += 1
                elif not s or n_citas >= CITAS_DIA_LLENO:
                    slots_info.append(f"{label}: DÍA CARGADO, no ofrecer proactivamente")
            fecha_cursor += timedelta(days=1)
            dias_revisados += 1
        fecha_ctx = "próximos días disponibles (PRIORIZAR hoy y mañana)"
        slots = slots_info

    # CHECK 1: Re-verificar handoff antes de generar respuesta (pudo activarse durante debounce/slots)
    state = await get_state(conversation_id)
    if state and state.handoff:
        log.info(f"[Conv {conversation_id}] HANDOFF detectado pre-generación (Redis) — IA NO responde")
        return
    cw_check1 = await get_custom_attributes(conversation_id)
    if cw_check1.get("pasar_supervisor") == "si":
        state.handoff = True
        state.handoff_at = state.handoff_at or datetime.now().isoformat()
        await save_state(state)
        log.info(f"[Conv {conversation_id}] HANDOFF detectado pre-generación (Chatwoot) — IA NO responde")
        return

    # Buscar citas existentes del paciente (por nombre o teléfono) para contexto
    citas_existentes = []
    if state.contact_name:
        citas_existentes = await appointments.buscar_citas_por_nombre(state.contact_name)
    if not citas_existentes and state.contact_phone:
        citas_existentes = await appointments.buscar_citas_por_telefono(state.contact_phone)

    # Generar respuesta con IA
    ai_response = await generate_response(
        state=state,
        user_message=combined_message,
        slots_disponibles=slots,
        fecha_contexto=fecha_ctx,
        citas_existentes=citas_existentes,
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

            if cita_id is None:
                # Slot ocupado — no enviar confirmación, escalar a supervisor
                log.warning(f"[Conv {conversation_id}] Slot {fecha} {hora} ocupado — escalando a supervisor")
                clean_text = "Disculpe, ese horario acaba de ser tomado. Una asesora le ayudará a encontrar otro horario disponible. Un momento por favor 😊"
                state.handoff = True
                state.handoff_at = datetime.now().isoformat()
                await save_state(state)
                await add_label(conversation_id, "supervisor")
                await set_custom_attributes(conversation_id, {"ai_status": "supervisor", "pasar_supervisor": "si"})
                await appointments.registrar_alerta(
                    tipo="supervisor", conversation_id=conversation_id,
                    contact_name=nombre, contact_phone=telefono,
                    detalle=f"Slot ocupado {fecha} {hora} — paciente necesita reagendar",
                )
                cita_data = None  # Limpiar para que no se marque como cita_creada
            else:
                log.info(f"[Conv {conversation_id}] CITA CREADA #{cita_id}: {nombre} {fecha} {hora}")

                # Registrar alerta de cita agendada
                await appointments.registrar_alerta(
                    tipo="cita_agendada",
                    conversation_id=conversation_id,
                    contact_name=nombre,
                    contact_phone=telefono,
                    detalle=f"Cita #{cita_id}: {fecha} {hora} — {cita_data.get('motivo', 'Consulta')}",
                )

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

                # Auto-detectar si el turno quedó lleno tras crear esta cita
                try:
                    gcal_busy_post = None
                    if gcal_token:
                        gcal_busy_post = await gcal.get_busy_slots(gcal_token, fecha)
                    auto_bloq = await appointments.detectar_y_bloquear_turno_lleno(fecha, gcal_busy_post)
                    if auto_bloq:
                        log.info(f"[Conv {conversation_id}] Turno(s) auto-bloqueado(s) tras cita: {auto_bloq}")
                except Exception as e:
                    log.error(f"Error en auto-detección turno lleno: {e}")

                await add_label(conversation_id, "cita_agendada")
                await set_custom_attributes(conversation_id, {"ai_status": "cita_agendada", "pasar_supervisor": "si"})
                # Handoff automático: IA deja de responder, equipo toma control
                state.cita_creada = True
                state.handoff = True
                state.handoff_at = datetime.now().isoformat()
                log.info(f"[Conv {conversation_id}] HANDOFF automático — cita creada, supervisor toma control")
                # Cerrar seguimiento para que no reciba follow-ups
                await appointments.cerrar_seguimiento(conversation_id)
                # Asignar a team en Chatwoot si está configurado
                if config.CHATWOOT_TEAM_ID:
                    await assign_team(conversation_id, config.CHATWOOT_TEAM_ID)
        except Exception as e:
            log.error(f"Error creando cita: {e}")
            # Si falla la creación, escalar a supervisor en vez de enviar confirmación falsa
            clean_text = "Hubo un inconveniente al registrar su cita. Una asesora le ayudará en un momento 😊"
            state.handoff = True
            state.handoff_at = datetime.now().isoformat()
            await save_state(state)
            await add_label(conversation_id, "supervisor")
            await set_custom_attributes(conversation_id, {"ai_status": "supervisor", "pasar_supervisor": "si"})
            cita_data = None

    # Detectar escalamiento a supervisor (IA decidió pasar a humano)
    supervisor_motivo = extract_supervisor_tag(ai_response)
    if supervisor_motivo and not state.handoff:
        state.handoff = True
        state.handoff_at = datetime.now().isoformat()
        log.info(f"[Conv {conversation_id}] HANDOFF por IA — motivo: {supervisor_motivo}")
        # Registrar alerta de supervisor
        await appointments.registrar_alerta(
            tipo="supervisor",
            conversation_id=conversation_id,
            contact_name=state.contact_name or contact_name or "",
            contact_phone=state.contact_phone or contact_phone or "",
            detalle=f"IA escalo: {supervisor_motivo}",
        )
        await add_label(conversation_id, "supervisor")
        await set_custom_attributes(conversation_id, {"ai_status": "supervisor", "pasar_supervisor": "si"})
        # Cerrar seguimiento para que no reciba follow-ups
        await appointments.cerrar_seguimiento(conversation_id)
        if config.CHATWOOT_TEAM_ID:
            await assign_team(conversation_id, config.CHATWOOT_TEAM_ID)
        # Nota privada para el equipo
        await send_message(conversation_id, f"🔔 IA escaló a supervisor: {supervisor_motivo}", private=True)

    # CHECK 2: Re-verificar handoff antes de enviar (pudo activarse mientras IA generaba respuesta)
    state_check = await get_state(conversation_id)
    cw_check2 = await get_custom_attributes(conversation_id)
    is_handoff_now = (state_check and state_check.handoff) or cw_check2.get("pasar_supervisor") == "si"
    if is_handoff_now and not cita_data and not supervisor_motivo:
        log.info(f"[Conv {conversation_id}] HANDOFF detectado pre-envío — mensaje IA descartado")
        return

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
    # NO crear seguimiento si la conversacion esta en handoff (supervisor tomó control)
    if not state.handoff:
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

    # Detectar cuando supervisor devuelve conversación a la IA (status → pending)
    if event == "conversation_status_changed":
        conversation = payload.get("conversation", payload)
        conv_id = conversation.get("id") or payload.get("id")
        new_status = conversation.get("status") or payload.get("status")
        if conv_id and new_status == "pending":
            state = await get_state(conv_id)
            if state and state.handoff:
                state.handoff = False
                state.handoff_at = None
                state.cita_creada = False
                await save_state(state)
                await set_custom_attributes(conv_id, {"ai_status": "ia_activa", "pasar_supervisor": "no"})
                log.info(f"[Conv {conv_id}] Supervisor devolvió conversación (status→pending) — IA retoma control")
                return {"ok": True, "action": "handoff_cleared"}
        return {"ok": True, "skipped": "status_change_no_action"}

    # Detectar cuando equipo cambia "Pasar supervisor" o labels desde Chatwoot
    if event == "conversation_updated":
        conversation = payload.get("conversation", payload)
        conv_id = conversation.get("id") or payload.get("id")
        changed = payload.get("changed_attributes", {})
        custom_attrs = conversation.get("custom_attributes", {})
        labels = conversation.get("labels", [])

        if not conv_id:
            return {"ok": True, "skipped": "conversation_updated_no_conv_id"}

        state = await get_state(conv_id)

        # --- Detectar cambio en LABELS (para app movil) ---
        label_changed = changed.get("labels") if isinstance(changed, dict) else None
        if label_changed:
            prev_labels = set(label_changed.get("previous_value", []))
            curr_labels = set(label_changed.get("current_value", labels))
            added_labels = curr_labels - prev_labels
            removed_labels = prev_labels - curr_labels
            log.info(f"[Conv {conv_id}] Labels changed: added={added_labels}, removed={removed_labels}")

            # Label "pasar_supervisor", "supervisor" o "cita_agendada" agregado → activar handoff
            if ("pasar_supervisor" in added_labels or "supervisor" in added_labels or "cita_agendada" in added_labels) and state and not state.handoff:
                state.handoff = True
                state.handoff_at = datetime.now().isoformat()
                state.cita_creada = "cita_agendada" in added_labels
                await save_state(state)
                ai_status = "cita_agendada" if "cita_agendada" in added_labels else "supervisor"
                await set_custom_attributes(conv_id, {"ai_status": ai_status, "pasar_supervisor": "si"})
                await appointments.cerrar_seguimiento(conv_id)
                await appointments.registrar_alerta(tipo="supervisor", conversation_id=conv_id, detalle="Handoff activado via label")
                log.info(f"[Conv {conv_id}] Handoff activado via LABEL — IA pausada")
                return {"ok": True, "action": "handoff_activated_via_label"}

            # Label "devolver_ia" agregado → desactivar handoff
            if "devolver_ia" in added_labels and state and state.handoff:
                state.handoff = False
                state.handoff_at = None
                state.cita_creada = False
                await save_state(state)
                await set_custom_attributes(conv_id, {"ai_status": "ia_activa", "pasar_supervisor": "no"})
                # Limpiar labels de handoff, dejar devolver_ia como registro
                await remove_label(conv_id, "pasar_supervisor")
                await remove_label(conv_id, "supervisor")
                await remove_label(conv_id, "devolver_ia")
                await add_label(conv_id, "ia_activa")
                log.info(f"[Conv {conv_id}] Handoff desactivado via LABEL 'devolver_ia' — IA retoma control")
                return {"ok": True, "action": "handoff_cleared_via_label"}

            # Label "pasar_supervisor" o "supervisor" removido → desactivar handoff
            if ("pasar_supervisor" in removed_labels or "supervisor" in removed_labels) and state and state.handoff:
                # Solo si no quedan labels de handoff activos
                if "pasar_supervisor" not in curr_labels and "supervisor" not in curr_labels:
                    state.handoff = False
                    state.handoff_at = None
                    state.cita_creada = False
                    await save_state(state)
                    await set_custom_attributes(conv_id, {"ai_status": "ia_activa", "pasar_supervisor": "no"})
                    log.info(f"[Conv {conv_id}] Handoff desactivado via remocion de LABEL — IA retoma control")
                    return {"ok": True, "action": "handoff_cleared_via_label_removal"}

        # --- Detectar cambio en custom attribute pasar_supervisor ---
        pasar = custom_attrs.get("pasar_supervisor")
        if pasar:
            if pasar == "si" and state and not state.handoff:
                state.handoff = True
                state.handoff_at = datetime.now().isoformat()
                await save_state(state)
                await set_custom_attributes(conv_id, {"ai_status": "supervisor", "pasar_supervisor": "si"})
                await appointments.cerrar_seguimiento(conv_id)
                log.info(f"[Conv {conv_id}] Equipo activó 'Pasar supervisor' — IA pausada")
                return {"ok": True, "action": "handoff_activated"}
            elif pasar == "no" and state and state.handoff:
                state.handoff = False
                state.handoff_at = None
                state.cita_creada = False
                await save_state(state)
                await set_custom_attributes(conv_id, {"ai_status": "ia_activa", "pasar_supervisor": "no"})
                log.info(f"[Conv {conv_id}] Equipo desactivó 'Pasar supervisor' — IA retoma control")
                return {"ok": True, "action": "handoff_cleared"}
        return {"ok": True, "skipped": "conversation_updated_no_action"}

    if not AI_ENABLED:
        log.info("[Webhook] IA pausada — ignorando mensaje")
        return {"ok": True, "skipped": "ai_disabled"}

    if event != "message_created":
        return {"ok": True, "skipped": "not_message_created"}

    message = payload.get("content", "")
    message_type = payload.get("message_type")

    if message_type != "incoming":
        # Auto-handoff: si una PERSONA del equipo responde manualmente (outgoing) y la IA no está en handoff
        # Excluir mensajes del bot (sender.id == BOT_CHATWOOT_USER_ID) porque usa el mismo token
        if message_type == "outgoing":
            sender_info = payload.get("sender", {})
            sender_type = sender_info.get("type", "")
            sender_id = sender_info.get("id")
            conv = payload.get("conversation", {})
            conv_id = conv.get("id")
            if conv_id and sender_type == "user" and sender_id != config.BOT_CHATWOOT_USER_ID:
                st = await get_state(conv_id)
                if st and not st.handoff:
                    st.handoff = True
                    st.handoff_at = datetime.now().isoformat()
                    await save_state(st)
                    await set_custom_attributes(conv_id, {"ai_status": "supervisor", "pasar_supervisor": "si"})
                    await add_label(conv_id, "supervisor")
                    log.info(f"[Conv {conv_id}] AUTO-HANDOFF: agente humano (id={sender_id}) respondió — IA pausada")
        return {"ok": True, "skipped": "not_incoming"}

    # Filtrar mensajes del equipo (ej: doctor escribiendo desde su numero personal)
    sender = payload.get("sender", {})
    sender_phone = sender.get("phone_number", "")
    if sender_phone:
        # Limpiar a solo últimos 9 dígitos para comparar
        sender_digits = ''.join(c for c in sender_phone if c.isdigit())[-9:]
        team_digits = {''.join(c for c in p if c.isdigit())[-9:] for p in config.TEAM_PHONES}
        if sender_digits in team_digits:
            log.info(f"[Webhook] Mensaje de teléfono del equipo ({sender_phone}) — IA ignora")
            return {"ok": True, "skipped": "team_phone"}

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
            tipo_paciente=body.get("tipo_paciente"),
            notas_equipo=body.get("notas", ""),
        )
        # Crear cita manual — bypass de validaciones de turno/conversación
        pool = await appointments.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO citas (nombre_paciente, telefono, fecha, hora, motivo, canal, estado, tipo_paciente, notas_equipo)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            """, cita.nombre_paciente, cita.telefono, cita.fecha, cita.hora,
                cita.motivo, cita.canal.value, cita.estado.value,
                cita.tipo_paciente, cita.notas_equipo)
            cita_id = row["id"]
        log.info(f"Cita manual #{cita_id}: {cita.nombre_paciente} {fecha} {hora} (estado={cita.estado.value})")

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
        for field in ("nombre_paciente", "telefono", "motivo", "notas_equipo", "tipo_paciente"):
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


@app.delete("/api/citas/{cita_id}")
async def eliminar_cita(cita_id: int, request: Request):
    """Eliminar una cita desde el dashboard (soft delete → cancelada)."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.decode_token(token)
    if not payload:
        raise HTTPException(401, "Token invalido")
    cita = await appointments.get_cita(cita_id)
    if not cita:
        raise HTTPException(404, "Cita no encontrada")
    await appointments.actualizar_estado(cita_id, EstadoCita.CANCELADA)
    # Eliminar de Google Calendar si tiene evento
    gcal_token = await appointments.get_gcal_token()
    if gcal_token and cita.get("gcal_event_id"):
        await gcal.delete_event(gcal_token, cita["gcal_event_id"])
    # Auto-desbloquear turno si la cancelación libera cupos
    turno = "manana" if cita["hora"].hour < 12 else "tarde"
    if await appointments.is_turno_bloqueado(cita["fecha"], turno):
        await appointments.desbloquear_turno(cita["fecha"], turno)
        log.info(f"Turno {turno} {cita['fecha']} auto-desbloqueado tras cancelación de cita #{cita_id}")
    log.info(f"Cita #{cita_id} eliminada (cancelada) por {payload.get('username', '?')}")
    return {"ok": True, "cita_id": cita_id}


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


# ─── Turnos bloqueados (manana/tarde por día) ───

@app.get("/api/turnos-bloqueados")
async def get_turnos_bloqueados(fecha: str = None):
    """Lista turnos bloqueados. Opcional: filtrar por fecha."""
    f = date.fromisoformat(fecha) if fecha else None
    turnos = await appointments.get_turnos_bloqueados(f)
    return {"turnos": turnos}


@app.post("/api/turnos-bloqueados")
async def bloquear_turno(request: Request):
    """Bloquear un turno (manana/tarde) de un día."""
    body = await request.json()
    fecha = date.fromisoformat(body["fecha"])
    turno = body["turno"]  # "manana" o "tarde"
    if turno not in ("manana", "tarde"):
        return JSONResponse({"error": "turno debe ser 'manana' o 'tarde'"}, status_code=400)
    motivo = body.get("motivo", "sin cupos")
    await appointments.bloquear_turno(fecha, turno, motivo)
    return {"ok": True, "fecha": body["fecha"], "turno": turno, "motivo": motivo}


@app.delete("/api/turnos-bloqueados")
async def desbloquear_turno(request: Request):
    """Desbloquear un turno de un día."""
    body = await request.json()
    fecha = date.fromisoformat(body["fecha"])
    turno = body["turno"]
    await appointments.desbloquear_turno(fecha, turno)
    return {"ok": True, "fecha": body["fecha"], "turno": turno}


# ─── Slots bloqueados (Ocupado Rápido) ───

@app.post("/api/slots/bloquear")
async def bloquear_slot(request: Request):
    """Bloquear un slot individual (crea cita con estado 'bloqueado')."""
    body = await request.json()
    fecha = date.fromisoformat(body["fecha"])
    hora_parts = body["hora"].split(":")
    from datetime import time as dt_time
    hora = dt_time(int(hora_parts[0]), int(hora_parts[1]))
    pool = await appointments.get_pool()
    async with pool.acquire() as conn:
        # Verificar si ya existe una cita en ese horario
        existing = await conn.fetchval(
            "SELECT id FROM citas WHERE fecha = $1 AND hora = $2 AND estado != 'cancelada'",
            fecha, hora
        )
        if existing:
            return {"ok": False, "detail": "Slot ya ocupado"}
        cita_id = await conn.fetchval("""
            INSERT INTO citas (nombre_paciente, telefono, fecha, hora, motivo, canal, estado)
            VALUES ('OCUPADO', '', $1, $2, 'Bloqueado por equipo', 'web', 'bloqueado')
            RETURNING id
        """, fecha, hora)
    return {"ok": True, "id": cita_id}


@app.delete("/api/slots/bloquear")
async def desbloquear_slot(request: Request):
    """Desbloquear un slot (elimina cita 'bloqueado')."""
    body = await request.json()
    fecha = date.fromisoformat(body["fecha"])
    hora_parts = body["hora"].split(":")
    from datetime import time as dt_time
    hora = dt_time(int(hora_parts[0]), int(hora_parts[1]))
    pool = await appointments.get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM citas WHERE fecha = $1 AND hora = $2 AND estado = 'bloqueado'",
            fecha, hora
        )
    return {"ok": True}


# ─── Handoff API ───

@app.post("/api/conversations/{conversation_id}/handoff")
async def handoff_conversation(conversation_id: int, request: Request):
    """Activa handoff: la IA deja de responder y el supervisor toma control."""
    body = await request.json() if await request.body() else {}
    activate = body.get("activate", True)  # True = handoff ON, False = devolver a IA

    state = await get_state(conversation_id)
    if state is None:
        state = ConversationState(
            contact_id=0,
            conversation_id=conversation_id,
            handoff=activate,
            handoff_at=datetime.now().isoformat() if activate else None,
        )
    else:
        state.handoff = activate
        state.handoff_at = datetime.now().isoformat() if activate else None

    await save_state(state)

    action = "ACTIVADO" if activate else "DESACTIVADO"
    log.info(f"[Conv {conversation_id}] Handoff {action} manualmente")

    # Registrar alerta si se activa handoff manual
    if activate:
        await appointments.registrar_alerta(
            tipo="supervisor",
            conversation_id=conversation_id,
            contact_name="",
            detalle="Handoff manual activado",
        )

    # Actualizar custom attribute en Chatwoot
    await set_custom_attributes(conversation_id, {
        "ai_status": "supervisor" if activate else "ia_activa",
        "pasar_supervisor": "si" if activate else "no",
    })

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


# ─── Alertas API ───

@app.get("/api/alertas")
async def get_alertas(limit: int = 50, no_leidas: bool = False):
    """Lista alertas recientes (handoff + citas agendadas)."""
    rows = await appointments.get_alertas(limit, solo_no_leidas=no_leidas)
    result = []
    for r in rows:
        item = dict(r)
        if item.get("created_at"):
            item["created_at"] = item["created_at"].isoformat()
        result.append(item)
    count = await appointments.contar_alertas_no_leidas()
    return {"alertas": result, "no_leidas": count}


@app.post("/api/alertas/leer")
async def marcar_alertas_leidas(request: Request):
    """Marca alertas como leidas."""
    body = await request.json() if await request.body() else {}
    ids = body.get("ids")  # None = marcar todas
    await appointments.marcar_alertas_leidas(ids)
    return {"ok": True}


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
