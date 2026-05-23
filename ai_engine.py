"""Motor de IA — Claude API para generar respuestas del agente."""

import json
import logging
from datetime import date, datetime
import anthropic
from config import ANTHROPIC_API_KEY, AI_MODEL, AI_MAX_TOKENS, TIMEZONE
from models import ConversationState

log = logging.getLogger(__name__)

client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """Eres la asistente virtual de la Clínica Respira Vida. Tu ÚNICO objetivo es agendar citas con el Dr. Hebert Cuenca, Neumólogo.

REGLAS ESTRICTAS:
- Sé BREVE. Máximo 2-3 oraciones por mensaje. No des párrafos largos.
- SIEMPRE redirige la conversación hacia agendar una cita.
- NUNCA des diagnósticos, recomendaciones médicas ni consejos clínicos.
- Si preguntan el costo de tratamientos, di que varía por paciente y que el doctor lo determina en consulta.
- Usa emojis con moderación (máximo 1-2 por mensaje).
- Habla de "usted" (formal pero cálido).
- Responde en español.

DATOS DE LA CLÍNICA:
- Doctor: Dr. Hebert Cuenca, Neumólogo
- Especialidad: Neumología y Alergias Respiratorias (NO alergias de piel)
- Dirección: Av. Arequipa 2050, Lince, Lima (altura CC Risso)
- Horario: Lunes a Sábado, 8:30am a 6:00pm
- Consulta: S/50
- Pruebas de laboratorio: S/100 a S/300 aprox
- Panel de alergias: S/170 (31 alérgenos + IgE total)
- Observación laboral: S/50
- Pagos: Efectivo, Yape, tarjetas, transferencias (presencial)
- Atiende niños desde 6 meses
- El paciente debe asistir presencialmente

NO REALIZAMOS: Prick Test, descarte TBC, consultas a domicilio, atención gestantes, alergias de piel.
TBC: Por normativa MINSA debe atenderse en centro MINSA o EsSalud.

FLUJO DE AGENDAMIENTO:
Cuando el paciente quiera agendar (o tú lo dirijas a eso):
1. Pregunta qué día prefiere (mañana o tarde)
2. Ofrece los horarios disponibles que te proporciono
3. Pide nombre completo y número de contacto
4. Confirma la cita con todos los datos

Cuando tengas TODOS los datos (nombre, teléfono, fecha, hora), responde con un JSON al final de tu mensaje así:
[CITA_JSON]{"nombre":"...","telefono":"...","fecha":"YYYY-MM-DD","hora":"HH:MM","motivo":"..."}[/CITA_JSON]

Este JSON es invisible para el paciente, solo lo usamos internamente. Tu mensaje de confirmación debe ser natural.

IMPORTANTE: Si el paciente te da su nombre y teléfono, o si ya los tienes del contacto, no los pidas de nuevo."""


async def generate_response(
    state: ConversationState,
    user_message: str,
    slots_disponibles: list[str] | None = None,
    fecha_contexto: str | None = None,
) -> str:
    """Genera respuesta del agente usando Claude."""

    # Construir contexto adicional
    context_parts = []
    hoy = datetime.now().strftime("%A %d de %B de %Y")
    context_parts.append(f"Fecha actual: {hoy}")

    if state.contact_name:
        context_parts.append(f"Nombre del contacto: {state.contact_name}")
    if state.contact_phone:
        context_parts.append(f"Teléfono del contacto: {state.contact_phone}")

    if slots_disponibles and fecha_contexto:
        context_parts.append(
            f"Horarios disponibles para {fecha_contexto}: {', '.join(slots_disponibles)}"
        )
    elif slots_disponibles is not None and len(slots_disponibles) == 0:
        context_parts.append(
            f"NO hay horarios disponibles para {fecha_contexto}. Sugiere otro día."
        )

    if state.nombre_capturado:
        context_parts.append(f"Nombre capturado: {state.nombre_capturado}")
    if state.telefono_capturado:
        context_parts.append(f"Teléfono capturado: {state.telefono_capturado}")
    if state.fecha_elegida:
        context_parts.append(f"Fecha elegida: {state.fecha_elegida}")
    if state.hora_elegida:
        context_parts.append(f"Hora elegida: {state.hora_elegida}")

    context_block = "\n".join(context_parts)

    # Construir mensajes
    messages = []
    for msg in state.messages[-8:]:  # Ultimos 8 mensajes de contexto
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    system = f"{SYSTEM_PROMPT}\n\nCONTEXTO ACTUAL:\n{context_block}"

    try:
        response = await client.messages.create(
            model=AI_MODEL,
            max_tokens=AI_MAX_TOKENS,
            system=system,
            messages=messages,
        )
        return response.content[0].text
    except Exception as e:
        log.error(f"Error Claude API: {e}")
        return "Disculpe, tuvimos un inconveniente. ¿En qué puedo ayudarle? 🏥"


def extract_appointment_json(text: str) -> dict | None:
    """Extrae el JSON de cita si existe en la respuesta."""
    start = text.find("[CITA_JSON]")
    end = text.find("[/CITA_JSON]")
    if start != -1 and end != -1:
        json_str = text[start + len("[CITA_JSON]"):end].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            log.warning(f"JSON de cita invalido: {json_str}")
    return None


def clean_response(text: str) -> str:
    """Limpia la respuesta removiendo el JSON interno."""
    start = text.find("[CITA_JSON]")
    end = text.find("[/CITA_JSON]")
    if start != -1 and end != -1:
        text = text[:start].strip() + text[end + len("[/CITA_JSON]"):].strip()
    return text.strip()
