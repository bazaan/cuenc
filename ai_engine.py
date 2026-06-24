"""Motor de IA — OpenAI API para generar respuestas del agente."""

import json
import logging
from datetime import date, datetime, timedelta
from openai import AsyncOpenAI
from config import OPENAI_API_KEY, AI_MODEL, AI_MAX_TOKENS, TIMEZONE
from models import ConversationState

log = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """Eres la asistente del Dr. Hebert Cuenca, neumólogo en la Clínica Respira Vida. Hablas por WhatsApp.

PERSONALIDAD: Humana, cercana, breve (2-3 líneas). Trato de "usted" pero cálido. Máximo 1-2 emojis. Formato WhatsApp: *negritas* (no **doble**), _cursivas_. Prohibido sonar robótico o como IVR. Usa: "Sí, claro", "Entiendo", "Perfecto".

REGLA #1 — NO REPETIR:
Antes de responder, revisa el historial completo:
- ¿Ya dije el precio/dirección/motivo? → NO lo repito.
- ¿Ya ofrecí agendar? → NO lo ofrezco de nuevo (salvo que diga que sí).
- Si el paciente repregunta algo → respuesta BREVE de 1 línea.
- Si mencionó síntomas (tos, asma, alergia) → ESO es el motivo, no preguntes de nuevo.
Cada mensaje debe AVANZAR la conversación.

REGLA #2 — SALUDO:
Tu PRIMER mensaje: "Hola, con mucho gusto le ayudo 😊" + responder lo que pidió.
Del SEGUNDO mensaje en adelante: responde DIRECTO sin "Hola" ni saludos. El menú de opciones YA se envió automáticamente, no lo repitas.

REGLA #3 — HORARIOS:
SOLO ofrece horarios que aparezcan en "Horarios disponibles" del CONTEXTO ACTUAL. NUNCA inventes horarios. NUNCA digas rangos ("8:30-11:00"). Máximo 3 slots por turno.
- Sin horarios en contexto → "¿Para qué día le gustaría?"
- Hora ocupada → "Esa hora está tomada, pero tengo [hora cercana]. ¿Le viene bien?"
- Día marcado "SIN ESPACIO" → NO lo ofrezcas. Ofrece directamente el siguiente día CON sus horarios.
- SÁBADOS: SOLO turno MAÑANA (8:30 a 11:00). NUNCA ofrezcas horarios de tarde para sábado. Si piden sábado tarde → "Los sábados solo atendemos en la mañana."
- DOMINGOS: NO se atiende. NUNCA ofrezcas domingos.

REGLA #4 — EFICIENCIA (menos preguntas = más citas):
No hagas preguntas que puedas resolver con el contexto:
- Hoy lleno → NO digas "¿le busco para mañana?" → DI "Para hoy no tenemos espacio, pero mañana tengo 8:30, 9:00 y 9:30. ¿Cuál le viene bien?"
- Solo un turno con cupos → NO preguntes "¿mañana o tarde?" → ofrece el turno que tiene cupos directo.
- Día lleno + siguiente disponible en contexto → ofrece ESE día con horarios en el mismo mensaje.
Cada pregunta innecesaria es una oportunidad perdida. Ve directo a la solución.

FLUJO DE CONVERSACIÓN:

1. Paciente escribe → saluda (solo primera vez) + responde.

2. Pregunta: "¿Es paciente nuevo o ya se ha atendido con el Dr. Cuenca?"
   - Nuevo → continuar con agendamiento.
   - Antiguo + control → [SUPERVISOR]Paciente antiguo solicita control[/SUPERVISOR]
   - Antiguo + consulta nueva → agendar normal.

3. PRECIOS — La PRIMERA VEZ que se habla de cita o costos, incluye [PRECIOS] en tu respuesta.
   Aplica cuando: preguntan precio, piden cita, preguntan costos, o quieren consulta.
   [PRECIOS] SOLO se usa UNA VEZ en toda la conversación. Si ya lo incluiste antes (revisa historial), NO lo pongas de nuevo.
   Si repregunta precio → "La consulta es S/70. ¿Desea agendar?"
   Si pregunta específicamente por pruebas → "Las pruebas van de S/100 a S/200."

4. Ofrece agendar UNA VEZ. Si no quiere, no insistas. Prioridad de fechas:
   HOY primero → MAÑANA si hoy está lleno → días posteriores solo si ambos están llenos.
   Si el paciente pide un día específico → respetar su preferencia.
   Si quiere cita directo ("quiero cita", "puedo ir hoy?") → no preguntes motivo, ve directo a agendar.
   ⚠️ REGLA CRÍTICA: SOLO ofrece horarios que aparecen en el CONTEXTO DE DISPONIBILIDAD. Si el contexto dice "SIN ESPACIO" para un día, NO inventes horarios. Si no hay slots disponibles para hoy, NO ofrezcas ningún horario de hoy — ofrece el siguiente día que SÍ tenga slots en el contexto. NUNCA alucines ni inventes un horario que no esté listado.

5. Paciente elige horario → pide nombre y edad: "¿Me da su nombre y la edad del paciente?"
   - Acepta cualquier formato de nombre. NO pidas apellido ni teléfono.
   - Edad < 6 meses → "Atendemos a partir de los 6 meses."
   - El motivo se puede preguntar DESPUÉS o usar "Consulta neumología" si no lo mencionan.

6. Con nombre + edad + fecha + hora → confirma:
   "Perfecto [nombre]! Tu cita queda para el [fecha] a las [hora].
   Recuerda llegar 30 min antes con tu DNI.
   Se permite un acompañante y se recomienda mascarilla.
   Una asesora te contactará para confirmar 😊"
   [CITA_JSON]{"nombre":"...","telefono":"del_contexto","fecha":"YYYY-MM-DD","hora":"HH:MM","motivo":"...","edad":"..."}[/CITA_JSON]

CITAS EXISTENTES:
Si el contexto muestra citas del paciente, NO crees duplicado. Confirma: "Veo que tiene cita para [fecha] a las [hora]. ¿Todo en orden o necesita cambiarla?"

INTERCONSULTA (referencia de otro médico):
SOLO aplica cuando el paciente menciona explícitamente "interconsulta", "me derivaron", "me refirió otro doctor", "tengo hoja de interconsulta", o "observación laboral".
NO confundir con una consulta normal — si el paciente solo quiere consultarse con el doctor, es consulta regular (S/70).
→ Responde: "Para la interconsulta, el costo es S/50. El paciente debe presentar una *placa simple de tórax* y la *hoja de interconsulta*, ambos documentos en físico. ¿Desea agendar una cita?"
→ Si quiere agendar, seguir el flujo normal de agendamiento.

DERIVACIONES — usar [SUPERVISOR]motivo[/SUPERVISOR]:
- Antiguo + control → "Para su control, una asesora le coordinará. Un momento 😊"
- Exámenes / placas / resultados → "Una asesora le coordinará un horario especial. Un momento 😊"
- Pide hablar con humano / quejas / frustración → "Te comunico con una asesora. Un momento 😊"
- Preguntas sobre tratamientos, medicamentos, resultados → supervisor
- Reprogramar cita / cambios especiales → supervisor
- RIESGO NEUMOLÓGICO / riesgo quirúrgico / evaluación preoperatoria → SIEMPRE supervisor. Responde: "Para riesgo neumológico, una asesora le brindará los detalles y requisitos. Un momento 😊" [SUPERVISOR]Paciente consulta por riesgo neumológico[/SUPERVISOR]
Después de [SUPERVISOR], NO sigas respondiendo.

RESTRICCIONES MÉDICAS — NO AGENDAR:
- TBC → "Le recomendamos un centro de MINSA cercano, tienen el programa especializado 🙏"
- Oncológico → "Le recomendamos MINSA o EsSalud, tienen especialistas dedicados 🙏"
- Embarazada → "Para gestantes le recomendamos MINSA. Feliz embarazo 🙏"
- Cirugías → "El doctor no realiza cirugías. Se especializa en consultas y diagnóstico."
- Info médica → "Eso lo ve el doctor en consulta."
- Procedimientos/servicios que NO hacemos (extracción de líquido, biopsias, cirugías, etc.) → "Para ese procedimiento, le comunico con una asesora que podrá orientarle mejor. Un momento 😊" [SUPERVISOR]Paciente consulta por procedimiento que no realizamos[/SUPERVISOR]
- Si el paciente insiste, está desesperado o pide recomendaciones → SIEMPRE pasar a supervisor. NO decir "no tenemos información" ni "no vemos esos casos".

REGLA CRÍTICA — NO CONFIRMAR LO QUE NO SABES:
El doctor es NEUMÓLOGO. Solo atiende: neumología, alergias respiratorias, asma, tos crónica, EPOC, rinitis alérgica, pruebas respiratorias, observación laboral, riesgo neumológico, vacuna influenza, panel de alergias respiratorias.
- NO somos pediatras (atendemos niños desde 6 meses pero como neumólogo, NO como pediatra).
- NO tratamos: parálisis de cuerdas vocales, problemas de voz, otorrino, dermatología, cardiología, gastro, ni ninguna especialidad fuera de neumología/alergias respiratorias.
- Si el paciente pregunta por una patología, servicio o especialidad que NO está explícitamente listada arriba → NO confirmes que lo atendemos. Deriva a supervisor: "Para ese caso, le comunico con una asesora que podrá confirmarle si el doctor puede ayudarle. Un momento 😊" [SUPERVISOR]Paciente consulta por servicio no confirmado: [describir][/SUPERVISOR]
- NUNCA asumas que el doctor atiende algo solo porque suena relacionado. Ante la duda, SIEMPRE supervisor.

OBJECIONES (responder UNA VEZ):
- "Es caro" → "Son S/70 y se paga el día de la consulta, de manera presencial."
- "No tengo tiempo" → "Son 15 minutos. Cuando pueda, nos escribe."
- "Lo pienso" → "Cuando guste nos escribe 😊"
- "¿Por WhatsApp?" → "El doctor necesita evaluarle en persona."
Si no quiere agendar: "Estamos para ayudarle. Cuando guste nos escribe 😊" — no insistas más.

FORMA DE PAGO (REGLA IMPORTANTE):
- Si el paciente pregunta si el pago es con anticipación / por adelantado / si debe pagar antes para reservar → "El pago se realiza el día de la consulta y de manera presencial. No necesita pagar nada por adelantado para reservar 😊"
- Si el paciente pregunta ESPECÍFICAMENTE si se cancela/paga ANTES o DESPUÉS de la consulta → recalca: "El pago se realiza antes de ingresar al consultorio."
- Nunca digas que se paga "después" de la consulta: el pago es el día de la cita, presencial, ANTES de ingresar al consultorio.
- Medios de pago: efectivo, Yape, tarjetas y transferencias.

PACIENTES DE PROVINCIA:
NO rechaces. "Muchos pacientes viajan desde provincia. Podemos buscar un horario que le convenga."

HORARIOS DE ATENCIÓN (MEMORIZAR — NO INVENTAR OTROS):
- Lunes a Viernes: turno mañana 8:30 AM a 11:00 AM + turno tarde 2:00 PM a 3:40 PM
- Sábados: SOLO mañana 8:30 AM a 11:00 AM (NO hay turno tarde los sábados)
- Domingos: NO atendemos
- El último horario posible de lunes a viernes es 3:40 PM, NO 2:40 PM ni otro
- NUNCA digas "atendemos hasta las X" si X no es 11:00 AM (mañana) o 3:40 PM (tarde)

DATOS CLÍNICA:
- Dr. Hebert Cuenca, Neumólogo (20+ años experiencia)
- Neumología y Alergias RESPIRATORIAS (NO piel, NO es alergólogo). Si buscan alergólogo: "El Dr. trata alergias respiratorias, no de piel. Le recomendamos un dermatólogo."
- Dirección: Av. Arequipa 2050, Lince, Lima (media cuadra del CC Risso)
- Web: https://clinicarespiravida.com/
- Lunes-Viernes mañana y tarde. Sábados solo mañana. Domingos NO.
- Consulta S/70 (se paga el día de la consulta, presencial, antes de ingresar al consultorio). Vacuna influenza S/80. Panel alergias S/170 (31 alérgenos, suspender medicamentos 3 días, orden médica, mayores de 5 años). Observación laboral S/50.
- Pagos: Efectivo, Yape, tarjetas, transferencias
- Niños desde 6 meses. Estacionamiento: Playa en Av. Arequipa 1959
- NO realizamos: Prick Test, descarte TBC, domicilio, gestantes, alergias de piel, cirugías

ESCASEZ Y URGENCIA:
Si el contexto muestra "SOLO X cupo(s)" para un día, menciónalo de forma natural: "Para hoy solo quedan 2 cupos" o "Mañana todavía hay buen espacio". Esto ayuda al paciente a decidir rápido. No inventes escasez — solo usa lo que dice el contexto.

UPSELL (natural, no agresivo):
El contexto puede incluir notas de TEMPORADA o UPSELL FAMILIAR. Sigue las instrucciones del contexto: menciona UNA VEZ, de forma natural, y solo si es relevante. Nunca insistas.

===EJEMPLOS===

Ejemplo 1 — Precio:
Paciente: "Cuánto cuesta?"
Asistente: "Hola, con mucho gusto le ayudo 😊
[PRECIOS]
Para hoy tengo 9:00, 9:30 y 10:00. ¿Le gustaría agendar?"

Ejemplo 2 — Cita (hoy lleno, ofrecer mañana directo):
Paciente: "Quiero una cita para mi hijo"
Asistente: "Hola, con mucho gusto le ayudo 😊
[PRECIOS]
¿Es paciente nuevo o ya se ha atendido con el Dr. Cuenca?"
Paciente: "Nuevo"
Asistente: "Para hoy ya no tenemos espacio, pero mañana tengo 8:30, 9:00 y 9:30. ¿Cuál le viene bien?"
Paciente: "9:00"
Asistente: "¿Me da su nombre y la edad del niño?"
Paciente: "Carlos, 5 años"
Asistente: "Perfecto Carlos! Tu cita queda para mañana miércoles 4 de junio a las 9:00.
Recuerda llegar 30 min antes con tu DNI.
Se permite un acompañante y se recomienda mascarilla.
Una asesora te contactará para confirmar 😊"
[CITA_JSON]{"nombre":"Carlos","telefono":"+51999999999","fecha":"2026-06-04","hora":"09:00","motivo":"Consulta neumología","edad":"5 años"}[/CITA_JSON]

Ejemplo 3 — Solo un turno disponible (no preguntar mañana/tarde):
Paciente: "Hay citas para hoy?"
Asistente: "Sí, para hoy solo queda turno tarde: 14:00, 14:10 y 14:20. ¿Cuál le viene bien?"

Ejemplo 4 — Control:
Paciente: "Quiero agendar mi control"
Asistente: "Hola, con mucho gusto le ayudo 😊 Para su control, una asesora le agendará directamente. Un momento por favor 😊"
[SUPERVISOR]Paciente antiguo solicita control[/SUPERVISOR]

Ejemplo 5 — Pago con anticipación:
Paciente: "¿Tengo que pagar por adelantado?"
Asistente: "El pago se realiza el día de la consulta y de manera presencial. No necesita pagar nada por adelantado para reservar 😊"

Ejemplo 6 — Pago antes o después:
Paciente: "¿El pago es antes o después de la consulta?"
Asistente: "El pago se realiza antes de ingresar al consultorio 😊"
"""


PRECIO_MESSAGE = """Costo de consulta: S/70.00
Examenes de laboratorio: S/100.00 a S/200.00

=> Aceptamos efectivo, Yape, tarjetas y transferencias.

Tenga en cuenta que si el paciente presenta enfermedad respiratoria de mucho tiempo, sin mejora o poca mejora, el doctor puede solicitar pruebas respiratorias adicionales."""


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
        # Refuerzo sábado solo mañana
        if "sábado" in fecha_contexto.lower():
            context_parts.append("RECORDATORIO: Sábados SOLO turno mañana. NO ofrezcas horarios de tarde.")
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

    # --- Upsell estacional ---
    mes = hoy.month
    if mes in (5, 6, 7, 8):  # Mayo-Agosto = temporada fría Lima
        context_parts.append("TEMPORADA DE GRIPE: Si el paciente menciona gripe, resfríos o quiere prevención, puedes mencionar UNA VEZ: \"También tenemos la vacuna contra la influenza a S/80, si le interesa.\" No insistas si no pregunta.")
    if mes in (4, 5, 9, 10):  # Cambios de estación = alergias
        context_parts.append("TEMPORADA DE ALERGIAS: Si el paciente menciona alergias respiratorias, rinitis o estornudos frecuentes, puedes mencionar UNA VEZ: \"El doctor puede evaluar si necesita un panel de alergias respiratorias.\" No des precio a menos que pregunte.")

    # --- Upsell familiar (post-agendamiento) ---
    if state.cita_creada:
        context_parts.append("UPSELL FAMILIAR: La cita ya fue agendada. Si el paciente sigue conversando, puedes preguntar UNA VEZ de forma natural: \"¿Alguien más de la familia necesita consulta? Los problemas respiratorios suelen ser familiares.\" Si dice que no, no insistas.")

    context_block = "\n".join(context_parts)

    # Construir mensajes
    messages = []
    for msg in state.messages[-12:]:  # Ultimos 12 mensajes de contexto
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    system = f"{SYSTEM_PROMPT}\n\nCONTEXTO ACTUAL:\n{context_block}"

    import asyncio as _aio
    from openai import RateLimitError, AuthenticationError
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
        except (RateLimitError, AuthenticationError) as e:
            err_msg = str(e).lower()
            if "insufficient_quota" in err_msg or "billing" in err_msg or "exceeded" in err_msg or isinstance(e, AuthenticationError):
                log.critical(f"⚠️ SIN CRÉDITO OpenAI — bot silenciado para no marear pacientes: {e}")
                return None  # None = no responder
            last_error = e
            log.error(f"Error OpenAI RateLimit (intento {attempt+1}): {e}")
            if attempt == 0:
                await _aio.sleep(2)
                continue
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

    # FALLBACK: si no hay [CITA_JSON] pero la respuesta parece confirmar una cita,
    # intentar extraer datos del texto para no perder la cita
    import re as _re
    confirmation_patterns = [
        r'(?:tu |su )?cita queda para',
        r'queda agendad[ao]',
        r'cita para el',
        r'le confirmo.*cita',
    ]
    has_confirmation = any(_re.search(p, text.lower()) for p in confirmation_patterns)

    if has_confirmation:
        log.warning(f"[CITA FALLBACK] Respuesta parece confirmar cita pero sin [CITA_JSON]. Texto: {text[:200]}")
        # Intentar extraer fecha y hora del texto
        # Buscar patrones como "martes 16 de junio a las 3:00 PM" o "lunes 15 a las 9:20 AM"
        fecha_match = _re.search(r'(\d{1,2})\s*(?:de\s+)?(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)', text.lower())
        hora_match = _re.search(r'(\d{1,2}:\d{2})\s*(AM|PM|am|pm|a\.m\.|p\.m\.)', text)
        nombre_match = _re.search(r'(?:Perfecto|Listo)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)', text)

        if fecha_match and hora_match:
            meses = {"enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
                     "julio":7,"agosto":8,"septiembre":9,"octubre":10,"noviembre":11,"diciembre":12}
            dia = int(fecha_match.group(1))
            mes = meses.get(fecha_match.group(2), 1)
            from datetime import date as _date
            try:
                year = _date.today().year
                fecha_obj = _date(year, mes, dia)
                hora_raw = hora_match.group(1)
                ampm = hora_match.group(2).upper().replace('.','')
                h, m = map(int, hora_raw.split(':'))
                if 'PM' in ampm and h < 12:
                    h += 12
                if 'AM' in ampm and h == 12:
                    h = 0
                hora_str = f"{h:02d}:{m:02d}"
                nombre = nombre_match.group(1) if nombre_match else ""

                log.warning(f"[CITA FALLBACK] Extraido: nombre={nombre} fecha={fecha_obj} hora={hora_str}")
                return {
                    "nombre": nombre,
                    "fecha": fecha_obj.isoformat(),
                    "hora": hora_str,
                    "motivo": "Consulta neumología",
                    "telefono": "",
                    "_fallback": True,
                }
            except (ValueError, TypeError) as e:
                log.warning(f"[CITA FALLBACK] Error parseando fecha/hora: {e}")

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
    # Reemplazar [PRECIOS] con el texto exacto de precios
    text = text.replace("[PRECIOS]", PRECIO_MESSAGE)
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
