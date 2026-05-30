"""Configuracion central del agente Doc C."""

import os

# --- Chatwoot ---
CHATWOOT_BASE_URL = os.getenv("CHATWOOT_BASE_URL", "https://clinicas.alefcompany.online")
CHATWOOT_API_TOKEN = os.getenv("CHATWOOT_API_TOKEN", "")
CHATWOOT_ACCOUNT_ID = os.getenv("CHATWOOT_ACCOUNT_ID", "1")
CHATWOOT_BOT_TOKEN = os.getenv("CHATWOOT_BOT_TOKEN", "")  # Agent Bot token (si se usa)

# --- OpenAI (Whisper transcripcion audio) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- AI (OpenAI) ---
AI_MODEL = os.getenv("AI_MODEL", "gpt-5.4-mini")
AI_MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", "5000"))  # Respuestas + JSON cita

# --- Redis (estado de conversacion) ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CONVERSATION_TTL = int(os.getenv("CONVERSATION_TTL", "3600"))  # 1h sin actividad = reset

# --- PostgreSQL (citas) ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://docc:docc_s3cur3@localhost:5432/docc_agent")

# --- App ---
PORT = int(os.getenv("PORT", "8090"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
JWT_SECRET = os.getenv("JWT_SECRET", "docc_jwt_s3cr3t_k3y_2024")
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "72"))
TIMEZONE = "America/Lima"

# --- Google Calendar ---
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "https://dashboard.respiravida.online/api/gcal/callback")

# --- Clinica ---
CLINICA_NOMBRE = "Clínica Respira Vida"
DOCTOR_NOMBRE = "Dr. Hebert Cuenca"
DOCTOR_ESPECIALIDAD = "Neumólogo"
CLINICA_TELEFONO = os.getenv("CLINICA_TELEFONO", "")  # Para derivar llamadas
CLINICA_DIRECCION = "Av. Arequipa 2050, Lince, Lima (altura CC Risso)"
HORARIO_INICIO = "08:30"
HORARIO_FIN = "16:00"  # legacy, no se usa directamente
HORARIO_FIN_SABADO = "12:00"  # legacy, no se usa directamente
# Turnos reales de agendamiento
TURNO_MANANA = ("08:30", "11:00")  # Lun-Sab
TURNO_TARDE = ("14:00", "15:40")   # Solo Lun-Vie
INTERVALO_CITAS_MIN = 10
SLOTS_VISIBLES = 3  # Mostrar solo 3 horarios (pedido del doctor)
CITAS_DIA_LLENO = 12  # Si un dia tiene >= 12 citas, no ofrecerlo proactivamente

# --- Handoff ---
CHATWOOT_TEAM_ID = int(os.getenv("CHATWOOT_TEAM_ID", "0"))  # Team "Asesoras" en Chatwoot para handoff

# Telefonos del equipo — la IA ignora mensajes de estos numeros
TEAM_PHONES = set(filter(None, os.getenv("TEAM_PHONES", "969460204").split(",")))

# ID del usuario Chatwoot cuyo token usa el bot — excluir del auto-handoff
# (mensajes outgoing del bot aparecen como sender_type=user con este ID)
BOT_CHATWOOT_USER_ID = int(os.getenv("BOT_CHATWOOT_USER_ID", "1"))
