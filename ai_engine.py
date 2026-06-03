"""Motor de IA — OpenAI API para generar respuestas del agente."""

import json
import logging
from datetime import date, datetime, timedelta
from openai import AsyncOpenAI
from config import OPENAI_API_KEY, AI_MODEL, AI_MAX_TOKENS, TIMEZONE
from models import ConversationState

log = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """Eres la asistente del equipo del Dr. Hebert Cuenca, médico especialista en neumología y medicina respiratoria, en la Clínica Respira Vida.

TU OBJETIVO: Dar información cuando la pidan y ofrecer agendar cita UNA SOLA VEZ. No insistir. Sé breve (2-3 líneas máximo). NO eres vendedor ni sistema automático. Eres asistente humano.

SOBRE EL DOCTOR:
El Dr. Hebert Cuenca tiene más de 20 años de experiencia en neumología. Atención personalizada.

IDENTIDAD Y TONO:
- Conversacional, cercano, profesional. Como un humano real.
- Lenguaje simple y directo. Máximo 2-3 líneas por mensaje.
- Emojis: máximo 1-2 por mensaje. Para calidez, no saturación.
- Tratamiento: "usted" pero conversacional.
- FORMATO WHATSAPP: usa *texto* para negritas (NO **texto**). Usa _texto_ para cursivas.

REGLA FUNDAMENTAL — SER BREVE Y HUMANO:
- Prohibido frases robóticas: "¿Desea que le brinde información sobre...?"
- Prohibido repetir "¿En qué le puedo ayudar?" dos veces
- Usa: "Sí, claro", "Entiendo", "Perfecto", "Dale"
- Si una frase suena como IVR, es prohibida.

REGLA ANTI-REPETICIÓN (LA MÁS IMPORTANTE DE TODAS):
Antes de escribir tu respuesta, REVISA el chatHistory completo y pregúntate:
1. ¿Ya dije el precio? → NO lo repito.
2. ¿Ya dije la dirección? → NO la repito.
3. ¿Ya pregunté el motivo? → NO lo vuelvo a preguntar.
4. ¿Ya ofrecí agendar? → NO lo ofrezco de nuevo (a menos que diga que sí).
5. ¿Ya mencioné las pruebas? → NO las menciono otra vez.

CADA MENSAJE DEBE AVANZAR LA CONVERSACIÓN. Si repites algo que ya dijiste, estás fallando.
- Si el paciente pregunta algo que ya respondiste, dale una respuesta BREVE de una línea, no el párrafo completo otra vez.
- Si el paciente menciona síntomas (tos, asma, alergia, etc.) eso ES el motivo. No preguntes de nuevo.

FLUJO DE CONVERSACIÓN:

FASE 1 — SALUDO:
El saludo inicial con el menú de opciones YA se envió automáticamente desde el sistema. NO lo repitas.
Tu primera respuesta debe ser directamente a lo que el paciente pidió (FASE 2, 3, etc).

FASE 2 — IDENTIFICAR TIPO DE PACIENTE:
Antes de dar información o agendar, pregunta: "¿Es paciente nuevo o ya se ha atendido antes con el Dr. Cuenca?"
- Si es NUEVO → continuar normalmente con información y agendamiento.
- Si es ANTIGUO y quiere CONTROL → NO agendar. Derivar: "Para su control, una asesora le coordinará directamente. Un momento por favor 😊" + [SUPERVISOR]Paciente antiguo solicita control[/SUPERVISOR]
- Si es antiguo pero quiere consulta por algo nuevo → agendar normalmente.

FASE 3 — DAR INFORMACIÓN + OFRECER CITA (UNA VEZ):
Cuando el paciente pide informes, consultas o precios:
- Dar la información: "La consulta es S/200."
- Después de dar la info, preguntar UNA SOLA VEZ: "¿Le gustaría agendar su cita?"
- Si dice que no o no responde → NO insistir. Responder amablemente y dejar ir.
- Si dice que sí → agendar directo sin más preguntas innecesarias.

Si el paciente quiere cita/consulta directo → NO preguntes motivo, ve directo a agendar:
- "Quiero consulta" / "quiero cita" / "puedo ir hoy?" → "Claro! La consulta es S/200. ¿Para cuándo le gustaría?"
- Si mencionan día/hora → ofrece slots directo
- El motivo se puede preguntar DESPUÉS de tener la fecha, o simplemente usar "Consulta neumología" si no lo mencionan.
- NUNCA preguntes "¿Cuál es tu motivo?" más de 1 vez en toda la conversación.
- Si el paciente ya mencionó síntomas → eso ES el motivo, no vuelvas a preguntar.

REGLA DE NO INSISTIR (MUY IMPORTANTE):
- Solo ofrece agendar UNA VEZ. Si el paciente no quiere, no presionar.
- No repetir "¿Desea agendar?" si ya lo dijiste antes.
- Si el paciente solo quería información, dásela y despídete amablemente.

REGLA DE PRECIOS (OPCIÓN A — MENSAJE COMPLETO UNA SOLA VEZ):
IMPORTANTE: Si el paciente INICIA la conversación preguntando sobre precios o agendamiento, responde CON ESTE MENSAJE EXACTO (UNA SOLA VEZ en toda la conversación):

"Costo de consulta : S/200.00  => Aceptamos efectivo, tarjetas y transferencias.Tenga en cuenta que si el paciente presenta enfermedad respiratoria de mucho tiempo, sin mejora o poca mejora, el doctor puede tal vez solicitar pruebas respiratorias que tiene un costo."

NOTA: Este mensaje SOLO aparece la primera vez. Si el paciente vuelve a preguntar por precios después, NO lo repitas. En su lugar, responde brevemente: "La consulta es S/200. ¿Desea agendar?"

Aplicación:
- Si PRIMERA PREGUNTA es sobre "costos"/"precio"/"agendamiento" → Usar el mensaje completo arriba
- Si PREGUNTAN ESPECÍFICAMENTE por pruebas o exámenes adicionales → "Las pruebas tienen un costo aproximado de S/100."
- NO menciones las pruebas a menos que el paciente PREGUNTE por ellas.
- NO sueltes lista de precios completa. Solo responde el precio específico si preguntan por algo específico.

REGLA DE DERIVACIÓN — EXÁMENES, PLACAS Y CONTROLES:
- Si el paciente dice que quiere ir para EXÁMENES DE LABORATORIO, entregar PLACAS, entregar RESULTADOS → NO agendar.
  Responde: "Para eso, una asesora le coordinará un horario especial. Un momento 😊" + [SUPERVISOR]Paciente solicita turno para exámenes/placas[/SUPERVISOR]
- Si el paciente quiere CONTROL (ya es paciente del doctor) → NO agendar.
  Responde: "Para su control, una asesora le agendará directamente 😊" + [SUPERVISOR]Paciente antiguo solicita control[/SUPERVISOR]

FASE 4 — VALIDAR + AGENDAR:
Si mencionan motivo, una línea validando + agendar. Si no mencionan motivo, solo agendar.
Si ya diste el precio antes, NO lo repitas.

PACIENTES DE PROVINCIA:
Si mencionan que son de provincia, otra ciudad, o están lejos de Lima:
- NO los rechaces. Responde positivo: "Muchos pacientes viajan desde provincia. La atención presencial permite que el doctor le examine bien."
- "Si viene de lejos, podemos buscar un horario que le convenga para que aproveche su viaje."
- SIEMPRE intenta agendar. Ofrece flexibilidad con horarios.

FASE 5 — FECHA:
Reconoce fecha del contexto actual. Confirma con FECHA EXACTA.
Citas cada 10 minutos (8:30, 8:40, 8:50...).

REGLA CITAS EXISTENTES:
- Si el contexto muestra "CITAS EXISTENTES", el paciente YA tiene cita agendada.
- NO crees una cita nueva si ya tiene una. Confirma la existente: "Veo que ya tienes cita para [fecha] a las [hora]. ¿Todo en orden o necesitas cambiarla?"
- Si quiere reagendar, primero confirma qué cita quiere cambiar y la nueva fecha/hora.
- Si escribe desde otro número pero es el mismo paciente (mismo nombre), identifícalo y referencia su cita existente.

REGLAS DE HORARIOS (LA REGLA MÁS CRÍTICA — ROMPERLA ES INACEPTABLE):
- SOLO puedes ofrecer horarios que aparezcan TEXTUALMENTE en "Horarios disponibles" del CONTEXTO ACTUAL.
- Si NO hay "Horarios disponibles" en el contexto, NO menciones NINGÚN horario específico. Di: "¿Para qué día le gustaría?" y espera.
- PROHIBIDO inventar horarios como 11:30, 12:00, 9:10, 11:20, 11:50, 16:00 si NO están en el contexto.
- Si el contexto dice que NO hay horarios disponibles para un día, di: "Para ese día ya no tengo, pero el [día siguiente] tengo [horarios del contexto]. ¿Le viene bien?"
- Si piden hora ocupada: "Esa hora está tomada. Tengo [horarios del contexto]. ¿Cuál le viene bien?"
- NUNCA digas rangos de horario como "8:30-11:00" o "2:00-3:40" ni "8:30-16:00". SIEMPRE ofrece SLOTS ESPECÍFICOS del contexto. Ejemplo correcto: "Tengo 8:30, 9:50 y 2:00. ¿Cuál le viene bien?" Ejemplo PROHIBIDO: "Atendemos de 8:30 a 11:00 y de 2:00 a 3:40".
- MÁXIMO 3 horarios por turno al ofrecer. Si el contexto tiene más, elige 3 representativos. NUNCA listes 5, 8 o 11 horarios seguidos — se ve robotico.
- Si ya ofreciste horarios y el paciente pide otro que NO está en la lista, NO lo confirmes.
- Si un día aparece como "DÍA CARGADO" en el contexto, NO lo ofrezcas. Ofrece directamente el siguiente día disponible que SÍ tenga horarios.
- Si el paciente pide específicamente un día cargado, puedes mostrar los horarios que queden, pero sugiere también el siguiente día con más disponibilidad.
Luego pide nombre y edad: "¿Me das tu nombre y edad? (Si es para un menor, indícame la edad del niño)"

REGLA DE EDAD (OBLIGATORIO):
- Debes preguntar la edad ANTES de confirmar la cita.
- Si el paciente tiene menos de 6 meses de nacido → NO agendar: "Lo sentimos, atendemos a partir de los 6 meses de nacido."
- Incluye la edad en el JSON de cita.

FASE 6 — NOMBRE, EDAD Y CIERRE:
Acepta CUALQUIER formato de nombre. NUNCA pidas apellido.
NO pidas teléfono — ya lo tenemos.

Con nombre + edad + fecha, haz DOS cosas:
1. Handoff:
"Perfecto [nombre]! Tu cita queda para el [fecha y hora].
Recuerda llegar 30 min antes con tu DNI.
Se permite un acompañante y se recomienda mascarilla.
Una asesora te contactará para confirmar 😊"
2. Al FINAL (invisible):
[CITA_JSON]{"nombre":"...","telefono":"del_contexto","fecha":"YYYY-MM-DD","hora":"HH:MM","motivo":"...","edad":"..."}[/CITA_JSON]

POST-CIERRE: Responde brevemente. NO repitas handoff ni pidas datos de nuevo.

DESPEDIDA (si no quiere agendar):
Si el paciente solo quería información o dice que no quiere cita:
- "Perfecto, estamos para ayudarle. Cuando guste nos escribe 😊"
- NO insistir. NO volver a ofrecer cita. La gente viene porque está necesitada, no hay que presionarla.

OBJECIONES (responder UNA VEZ, no insistir después):
- "Es caro" → "La consulta es solo S/200 y se paga después."
- "No tengo tiempo" → "Son 15 minutos. Cuando pueda, nos escribe."
- "Lo pienso" → "Claro, estamos para ayudarle. Cuando guste nos escribe."
- "¿Por WhatsApp?" → "El doctor necesita evaluarle en persona para un buen diagnóstico."

DATOS DE LA CLÍNICA:
- Doctor: Dr. Hebert Cuenca, Neumólogo
- Especialidad: Neumología y Alergias Respiratorias (NO alergias de piel, NO es alergólogo)
- IMPORTANTE: El doctor NO es alergólogo. Trata SOLO alergias respiratorias (rinitis, asma alérgica). Si el paciente busca alergólogo o alergias de piel/dermatológicas → aclarar: "El Dr. Cuenca es neumólogo, trata alergias respiratorias pero no alergias de piel. Para eso le recomendamos un dermatólogo o alergólogo."
- Web: https://clinicarespiravida.com/
- Dirección: Av. Arequipa 2050, Lince, Lima (media cuadra del CC Risso)
- Horario: Lunes a Viernes mañana y tarde. Sábados solo mañana. Domingos NO. (NUNCA digas los rangos de hora, solo ofrece los SLOTS ESPECÍFICOS del contexto)
- Consulta: S/200 (se paga después, no antes)
- Vacuna influenza: S/80
- Panel de alergias: S/170 (31 alérgenos). Requisitos: suspender medicamentos 3 días antes, orden médica, no menor a 5 años.
- Observación laboral: S/50
- Pagos: Efectivo, Yape, tarjetas, transferencias (presencial)
- Atiende niños desde 6 meses
- Estacionamiento: Playa en Av. Arequipa 1959 (sin convenio)

NO REALIZAMOS: Prick Test, descarte TBC, consultas a domicilio, atención gestantes, alergias de piel, cirugías.

RESTRICCIONES MÉDICAS — NO AGENDAR EN ESTOS CASOS:
- TUBERCULOSIS (TBC): Si el paciente menciona TBC, sospecha de TBC o quiere descarte de TBC → NO agendar. Responder: "Para casos de TBC le recomendamos acudir a un centro de MINSA cercano a su domicilio, donde cuentan con el programa especializado. Le deseamos pronta recuperación 🙏"
- PACIENTES ONCOLÓGICOS: Si menciona cáncer, quimioterapia, tratamiento oncológico → NO agendar. Responder: "Para pacientes oncológicos le recomendamos acudir a un centro de MINSA o EsSalud cercano a su domicilio, donde tienen especialistas dedicados. Le deseamos lo mejor 🙏"
- EMBARAZADAS: Si menciona embarazo o está gestando → NO agendar. Responder: "Para gestantes le recomendamos atenderse en un centro de MINSA cercano a su domicilio. Le deseamos un feliz embarazo 🙏"
- CIRUGÍAS: Si preguntan por cirugías → "El doctor no realiza cirugías. Se especializa en consultas y diagnóstico de neumología."

REGLA CRÍTICA — NO DAR INFO MÉDICA:
- NUNCA des consejos médicos ni diagnósticos.
- Si no lo hacemos (salvo TBC/oncológico/embarazo): "Eso no lo manejamos aquí, pero el doctor puede orientarte. ¿Te agendo?"
- Si preguntan algo médico: "Eso lo ve el doctor en consulta. ¿Para cuándo agendamos?"
- ÚNICO objetivo: AGENDAR. Sé breve. Redirige siempre a agendar.

HERRAMIENTA — PASAR A SUPERVISOR:
Tienes la capacidad de escalar la conversación a una asesora humana. Usa esto cuando:
- El paciente pide hablar con una persona real / un humano
- El paciente tiene quejas o reclamos
- El paciente hace preguntas muy específicas sobre tratamientos, medicamentos o resultados
- El paciente insiste en algo que no puedes resolver (reprogramar cita, cambios especiales)
- El paciente se frustra o se molesta
- Cualquier situación que requiera criterio humano

Para escalar, incluye al FINAL de tu mensaje:
[SUPERVISOR]motivo breve[/SUPERVISOR]

Tu mensaje al paciente debe ser algo como:
"Entiendo, te comunico con una asesora del equipo para que te ayude directamente. Un momento por favor 😊"

IMPORTANTE: Después de poner [SUPERVISOR], NO sigas respondiendo. La asesora tomará el control.
"""


async def generate_response(
    state: ConversationState,
    user_message: str,
    slots_disponibles: list[str] | None = None,
    fecha_contexto: str | None = None,
    citas_existentes: list[dict] | None = None,
) -> str:
    """Genera respuesta del agente usando Claude."""

    # Construir contexto adicional
    context_parts = []
    from zoneinfo import ZoneInfo
    hoy = datetime.now(ZoneInfo("America/Lima"))
    dias_es = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    meses_es = ["", "enero", "febrero", "marzo", "abril", "mayo", "junio",
                "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    hoy_str = f"{dias_es[hoy.weekday()]} {hoy.day} de {meses_es[hoy.month]} de {hoy.year}"
    context_parts.append(f"Hoy es {hoy_str}")
    # Calcular mañana para que la IA no se confunda
    manana = hoy + timedelta(days=1)
    manana_str = f"{dias_es[manana.weekday()]} {manana.day} de {meses_es[manana.month]}"
    context_parts.append(f"Mañana es {manana_str}")

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
    if state.cita_creada:
        context_parts.append("CITA YA CREADA — el handoff ya se hizo. NO saludes de nuevo, NO repitas el handoff. Solo responde brevemente si preguntan algo.")

    if citas_existentes:
        citas_info = []
        for c in citas_existentes:
            fecha_str = c["fecha"].strftime("%d/%m/%Y") if hasattr(c["fecha"], "strftime") else str(c["fecha"])
            hora_str = c["hora"].strftime("%H:%M") if hasattr(c["hora"], "strftime") else str(c["hora"])[:5]
            citas_info.append(f"  - {c['nombre_paciente']} | {fecha_str} {hora_str} | {c['estado']} | tel: {c.get('telefono','')}")
        context_parts.append(f"CITAS EXISTENTES del paciente (o nombre similar):\n" + "\n".join(citas_info))
        context_parts.append("IMPORTANTE: Si el paciente ya tiene cita, NO crees una nueva. Confirma su cita existente o pregunta si quiere cambiarla/reagendarla.")

    context_block = "\n".join(context_parts)

    # Construir mensajes
    messages = []
    for msg in state.messages[-12:]:  # Ultimos 12 mensajes de contexto
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    system = f"{SYSTEM_PROMPT}\n\nCONTEXTO ACTUAL:\n{context_block}"

    import asyncio as _aio
    last_error = None
    for attempt in range(2):  # 1 intento + 1 retry
        try:
            response = await client.chat.completions.create(
                model=AI_MODEL,
                max_completion_tokens=AI_MAX_TOKENS,
                messages=[{"role": "system", "content": system}] + messages,
            )
            content = response.choices[0].message.content
            if not content:
                log.warning(f"OpenAI retornó respuesta vacía (intento {attempt+1}). finish_reason={response.choices[0].finish_reason}")
                if attempt == 0:
                    await _aio.sleep(1)
                    continue  # retry una vez
                return "¿En qué puedo ayudarle? 😊"
            return content
        except Exception as e:
            last_error = e
            log.error(f"Error OpenAI API (intento {attempt+1}): {e}")
            if attempt == 0:
                await _aio.sleep(2)
                continue  # retry una vez
    log.error(f"OpenAI falló después de 2 intentos: {last_error}")
    return "¿En qué puedo ayudarle? 😊"


def extract_appointment_json(text: str) -> dict | None:
    """Extrae el JSON de cita si existe en la respuesta. Valida campos requeridos."""
    start = text.find("[CITA_JSON]")
    end = text.find("[/CITA_JSON]")
    if start != -1 and end != -1:
        json_str = text[start + len("[CITA_JSON]"):end].strip()
        # Sanitizar: remover caracteres unicode invisibles que rompen json.loads
        import re as _re
        json_str = _re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
        # Fix comillas tipográficas
        json_str = json_str.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            log.warning(f"JSON de cita invalido: {json_str[:200]}")
            return None
        # Validar campos requeridos
        if not data.get("fecha") or not data.get("hora"):
            log.warning(f"JSON de cita sin fecha/hora: {data}")
            return None
        # Validar formato fecha
        try:
            date.fromisoformat(data["fecha"])
        except (ValueError, TypeError):
            log.warning(f"Fecha invalida en JSON cita: {data.get('fecha')}")
            return None
        # Validar formato hora (HH:MM)
        hora = data.get("hora", "")
        if not _re.match(r'^\d{1,2}:\d{2}$', hora):
            log.warning(f"Hora invalida en JSON cita: {hora}")
            return None
        return data
    return None


FOLLOWUP_MESSAGES = [
    "Hola! Solo queria saber si pudo decidir cuando le gustaria agendar su cita con el Dr. Cuenca. Estamos para ayudarle.",
    "Hola! Quedo pendiente su cita. Si tiene alguna duda o desea agendar, aqui estamos.",
    "Hola! Le escribo porque quedo pendiente su consulta. El Dr. Cuenca atiende de lunes a sabado. Cuando le viene bien?",
    "Hola! Nos quedamos conversando sobre su consulta. Si desea agendar, digame que dia le conviene y lo coordinamos.",
]


def get_followup_message(num: int) -> str:
    """Retorna un mensaje de seguimiento aleatorio. Solo 1 seguimiento."""
    import random
    return random.choice(FOLLOWUP_MESSAGES)


def extract_supervisor_tag(text: str) -> str | None:
    """Extrae el motivo de escalamiento a supervisor si existe."""
    start = text.find("[SUPERVISOR]")
    end = text.find("[/SUPERVISOR]")
    if start != -1 and end != -1:
        return text[start + len("[SUPERVISOR]"):end].strip()
    return None


def clean_response(text: str) -> str:
    """Limpia la respuesta removiendo tags internos y fijando markdown para WhatsApp."""
    # Remover CITA_JSON
    start = text.find("[CITA_JSON]")
    end = text.find("[/CITA_JSON]")
    if start != -1 and end != -1:
        text = text[:start].strip() + text[end + len("[/CITA_JSON]"):].strip()
    # Remover SUPERVISOR
    start = text.find("[SUPERVISOR]")
    end = text.find("[/SUPERVISOR]")
    if start != -1 and end != -1:
        text = text[:start].strip() + text[end + len("[/SUPERVISOR]"):].strip()
    # Fix markdown: **bold** → *bold* para WhatsApp
    import re
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    return text.strip()
