"""Gestion de citas — base de datos PostgreSQL."""

import logging
from datetime import date, time, datetime, timedelta
from typing import Optional
import asyncpg
from config import DATABASE_URL, HORARIO_INICIO, HORARIO_FIN, INTERVALO_CITAS_MIN, SLOTS_VISIBLES
from models import Cita, EstadoCita, Canal

log = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    return _pool


async def init_db():
    """Crea las tablas si no existen."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS citas (
                id SERIAL PRIMARY KEY,
                nombre_paciente TEXT NOT NULL,
                telefono TEXT NOT NULL,
                fecha DATE NOT NULL,
                hora TIME NOT NULL,
                motivo TEXT,
                canal TEXT DEFAULT 'whatsapp',
                estado TEXT DEFAULT 'pendiente',
                conversation_id INTEGER,
                contact_id INTEGER,
                notas_equipo TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_citas_fecha ON citas(fecha);
            CREATE INDEX IF NOT EXISTS idx_citas_estado ON citas(estado);
        """)
        log.info("DB inicializada: tabla citas lista")


async def crear_cita(cita: Cita) -> int:
    """Crea una cita y retorna el ID."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO citas (nombre_paciente, telefono, fecha, hora, motivo, canal, estado, conversation_id, contact_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
        """, cita.nombre_paciente, cita.telefono, cita.fecha, cita.hora,
            cita.motivo, cita.canal.value, cita.estado.value,
            cita.conversation_id, cita.contact_id)
        log.info(f"Cita creada #{row['id']}: {cita.nombre_paciente} - {cita.fecha} {cita.hora}")
        return row["id"]


async def get_slots_ocupados(fecha: date) -> list[time]:
    """Retorna las horas ya agendadas para una fecha."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT hora FROM citas
            WHERE fecha = $1 AND estado NOT IN ('cancelada')
            ORDER BY hora
        """, fecha)
        return [r["hora"] for r in rows]


def generar_slots(fecha: date) -> list[time]:
    """Genera todos los slots posibles para un dia."""
    inicio = datetime.strptime(HORARIO_INICIO, "%H:%M")
    fin = datetime.strptime(HORARIO_FIN, "%H:%M")
    slots = []
    current = inicio
    while current < fin:
        slots.append(current.time())
        current += timedelta(minutes=INTERVALO_CITAS_MIN)
    return slots


async def get_slots_disponibles(fecha: date) -> list[str]:
    """
    Retorna slots disponibles formateados.
    Solo muestra SLOTS_VISIBLES opciones dispersas (pedido del doctor).
    """
    todos = generar_slots(fecha)
    ocupados = await get_slots_ocupados(fecha)
    disponibles = [s for s in todos if s not in ocupados]

    if not disponibles:
        return []

    # Dispersar: tomar slots equidistantes para que no se vea vacio
    if len(disponibles) <= SLOTS_VISIBLES:
        seleccion = disponibles
    else:
        step = len(disponibles) // SLOTS_VISIBLES
        seleccion = [disponibles[i * step] for i in range(SLOTS_VISIBLES)]

    return [s.strftime("%I:%M %p").lstrip("0") for s in seleccion]


async def get_citas_dia(fecha: date) -> list[dict]:
    """Retorna todas las citas de un dia (para el panel)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM citas
            WHERE fecha = $1
            ORDER BY hora ASC
        """, fecha)
        return [dict(r) for r in rows]


async def get_citas_rango(desde: date, hasta: date) -> list[dict]:
    """Retorna citas en un rango de fechas."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM citas
            WHERE fecha BETWEEN $1 AND $2
            ORDER BY fecha ASC, hora ASC
        """, desde, hasta)
        return [dict(r) for r in rows]


async def actualizar_estado(cita_id: int, estado: EstadoCita, notas: Optional[str] = None):
    """Actualiza el estado de una cita (desde el panel)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if notas:
            await conn.execute("""
                UPDATE citas SET estado = $1, notas_equipo = $2 WHERE id = $3
            """, estado.value, notas, cita_id)
        else:
            await conn.execute("""
                UPDATE citas SET estado = $1 WHERE id = $2
            """, estado.value, cita_id)


async def stats_dia(fecha: date) -> dict:
    """Estadisticas del dia."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM citas WHERE fecha = $1", fecha)
        confirmadas = await conn.fetchval(
            "SELECT COUNT(*) FROM citas WHERE fecha = $1 AND estado = 'confirmada'", fecha)
        pendientes = await conn.fetchval(
            "SELECT COUNT(*) FROM citas WHERE fecha = $1 AND estado = 'pendiente'", fecha)
        return {"total": total, "confirmadas": confirmadas, "pendientes": pendientes}
