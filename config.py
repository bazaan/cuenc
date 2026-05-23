"""Configuracion central del agente Doc C."""

import os

# --- Chatwoot ---
CHATWOOT_BASE_URL = os.getenv("CHATWOOT_BASE_URL", "https://clinicas.alefcompany.online")
CHATWOOT_API_TOKEN = os.getenv("CHATWOOT_API_TOKEN", "")
CHATWOOT_ACCOUNT_ID = os.getenv("CHATWOOT_ACCOUNT_ID", "1")
CHATWOOT_BOT_TOKEN = os.getenv("CHATWOOT_BOT_TOKEN", "")  # Agent Bot token (si se usa)

# --- Claude API ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "claude-sonnet-4-20250514")
AI_MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", "400"))  # Respuestas cortas + JSON cita

# --- Redis (estado de conversacion) ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CONVERSATION_TTL = int(os.getenv("CONVERSATION_TTL", "3600"))  # 1h sin actividad = reset

# --- PostgreSQL (citas) ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://docc:docc_s3cur3@localhost:5432/docc_agent")

# --- App ---
PORT = int(os.getenv("PORT", "8090"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
TIMEZONE = "America/Lima"

# --- Clinica ---
CLINICA_NOMBRE = "Clínica Respira Vida"
DOCTOR_NOMBRE = "Dr. Hebert Cuenca"
DOCTOR_ESPECIALIDAD = "Neumólogo"
CLINICA_TELEFONO = os.getenv("CLINICA_TELEFONO", "")  # Para derivar llamadas
CLINICA_DIRECCION = "Av. Arequipa 2050, Lince, Lima (altura CC Risso)"
HORARIO_INICIO = "08:30"
HORARIO_FIN = "18:00"
INTERVALO_CITAS_MIN = 15
SLOTS_VISIBLES = 3  # Mostrar solo 3 horarios (pedido del doctor)
