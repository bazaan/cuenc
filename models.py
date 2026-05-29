"""Modelos Pydantic para el agente."""

from datetime import datetime, date, time
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class EstadoCita(str, Enum):
    PENDIENTE = "pendiente"
    CONFIRMADA = "confirmada"
    CANCELADA = "cancelada"
    NO_CONTESTO = "no_contesto"
    ATENDIDA = "atendida"


class Canal(str, Enum):
    WHATSAPP = "whatsapp"
    INSTAGRAM = "instagram"
    MESSENGER = "messenger"
    TIKTOK = "tiktok"
    WEB = "web"


class TipoPaciente(str, Enum):
    NUEVO = "nuevo"
    PRIMER_CONTROL = "primer_control"
    ANTIGUO = "antiguo"
    PROCEDIMIENTO = "procedimiento"
    LEVANTAMIENTO = "levantamiento"


class Cita(BaseModel):
    id: Optional[int] = None
    nombre_paciente: str
    telefono: str
    fecha: date
    hora: time
    motivo: Optional[str] = None
    canal: Canal = Canal.WHATSAPP
    estado: EstadoCita = EstadoCita.PENDIENTE
    tipo_paciente: Optional[str] = None
    conversation_id: Optional[int] = None
    contact_id: Optional[int] = None
    notas_equipo: Optional[str] = None
    created_at: Optional[datetime] = None


class ConversationState(BaseModel):
    """Estado de la conversacion en curso (guardado en Redis)."""
    contact_id: int
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None
    conversation_id: int
    inbox_id: Optional[int] = None
    canal: Canal = Canal.WHATSAPP

    # Estado del flujo de agendamiento
    step: str = "inicio"  # inicio | pregunta_dia | pregunta_hora | pregunta_nombre | pregunta_telefono | confirmar | completado
    nombre_capturado: Optional[str] = None
    telefono_capturado: Optional[str] = None
    fecha_elegida: Optional[str] = None  # YYYY-MM-DD
    hora_elegida: Optional[str] = None   # HH:MM
    motivo: Optional[str] = None

    # Historial para contexto IA (ultimos N mensajes)
    messages: list[dict] = []
    cita_creada: bool = False
    handoff: bool = False  # True = supervisor tomó control, IA no responde
    handoff_at: Optional[str] = None  # ISO timestamp de cuando se activo handoff
