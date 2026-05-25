"""Gestion de citas — base de datos PostgreSQL."""

import json
import logging
from datetime import date, time, datetime, timedelta
from typing import Optional
import asyncpg
from config import DATABASE_URL, HORARIO_INICIO, HORARIO_FIN, HORARIO_FIN_SABADO, INTERVALO_CITAS_MIN, SLOTS_VISIBLES
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

            CREATE TABLE IF NOT EXISTS ejecuciones (
                id SERIAL PRIMARY KEY,
                conversation_id INTEGER,
                contact_name TEXT,
                contact_phone TEXT,
                canal TEXT,
                mensaje_usuario TEXT,
                respuesta_agente TEXT,
                tipo TEXT DEFAULT 'texto',
                cita_creada BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_ejecuciones_created ON ejecuciones(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_ejecuciones_conv ON ejecuciones(conversation_id);

            -- Agregar columna contact_phone si no existe (migracion)
            DO $$ BEGIN
                ALTER TABLE ejecuciones ADD COLUMN IF NOT EXISTS contact_phone TEXT;
            EXCEPTION WHEN OTHERS THEN NULL;
            END $$;

            CREATE TABLE IF NOT EXISTS seguimientos (
                id SERIAL PRIMARY KEY,
                conversation_id INTEGER NOT NULL,
                contact_name TEXT,
                ultimo_mensaje_at TIMESTAMPTZ NOT NULL,
                seguimiento_num INTEGER DEFAULT 0,
                seguimiento1_at TIMESTAMPTZ,
                seguimiento2_at TIMESTAMPTZ,
                cita_creada BOOLEAN DEFAULT FALSE,
                cerrado BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_seguimientos_conv
                ON seguimientos(conversation_id) WHERE NOT cerrado;

            -- Google Calendar tokens
            CREATE TABLE IF NOT EXISTS gcal_tokens (
                id SERIAL PRIMARY KEY,
                email TEXT DEFAULT '',
                token_data JSONB NOT NULL,
                is_primary BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );

            -- Agregar gcal_event_id a citas
            DO $$ BEGIN
                ALTER TABLE citas ADD COLUMN IF NOT EXISTS gcal_event_id TEXT;
            EXCEPTION WHEN OTHERS THEN NULL;
            END $$;

            -- Dias bloqueados (equipo marca dia como lleno)
            CREATE TABLE IF NOT EXISTS dias_bloqueados (
                fecha DATE PRIMARY KEY,
                motivo TEXT DEFAULT 'lleno',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        log.info("DB inicializada: tablas citas y ejecuciones listas")


async def crear_cita(cita: Cita) -> int:
    """Crea una cita y retorna el ID. Previene duplicados por conversation_id."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Verificar si ya existe una cita activa para esta conversación
        if cita.conversation_id:
            existing = await conn.fetchval("""
                SELECT id FROM citas
                WHERE conversation_id = $1 AND estado NOT IN ('cancelada')
                LIMIT 1
            """, cita.conversation_id)
            if existing:
                log.warning(f"Cita duplicada bloqueada: conv {cita.conversation_id} ya tiene cita #{existing}")
                return existing

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


RECESO_INICIO = time(13, 0)  # 1:00 PM
RECESO_FIN = time(14, 0)     # 2:00 PM


def generar_slots(fecha: date) -> list[time]:
    """Genera todos los slots posibles para un dia. Sabado termina a mediodia. Excluye receso 1-2pm."""
    inicio = datetime.strptime(HORARIO_INICIO, "%H:%M")
    horario_fin = HORARIO_FIN_SABADO if fecha.weekday() == 5 else HORARIO_FIN
    fin = datetime.strptime(horario_fin, "%H:%M")
    slots = []
    current = inicio
    while current < fin:
        t = current.time()
        # Excluir receso 1pm-2pm
        if not (RECESO_INICIO <= t < RECESO_FIN):
            slots.append(t)
        current += timedelta(minutes=INTERVALO_CITAS_MIN)
    return slots


async def is_dia_bloqueado(fecha: date) -> bool:
    """Verifica si un día está bloqueado por el equipo."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchval("SELECT 1 FROM dias_bloqueados WHERE fecha = $1", fecha)
        return row is not None


async def bloquear_dia(fecha: date, motivo: str = "lleno"):
    """Bloquea un día (no se ofrecen slots)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO dias_bloqueados (fecha, motivo) VALUES ($1, $2)
            ON CONFLICT (fecha) DO UPDATE SET motivo = $2
        """, fecha, motivo)
    log.info(f"Día bloqueado: {fecha} ({motivo})")


async def desbloquear_dia(fecha: date):
    """Desbloquea un día."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM dias_bloqueados WHERE fecha = $1", fecha)
    log.info(f"Día desbloqueado: {fecha}")


async def get_dias_bloqueados() -> list[dict]:
    """Lista todos los días bloqueados."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT fecha, motivo FROM dias_bloqueados ORDER BY fecha")
        return [{"fecha": r["fecha"].isoformat(), "motivo": r["motivo"]} for r in rows]


async def get_slots_disponibles(fecha: date, gcal_busy: list | None = None) -> list[str]:
    """
    Retorna slots disponibles formateados.
    Solo muestra SLOTS_VISIBLES opciones dispersas (pedido del doctor).
    gcal_busy: lista de tuplas (start_dt, end_dt) de periodos ocupados en GCal.
    """
    # Si el día está bloqueado, no hay slots
    if await is_dia_bloqueado(fecha):
        return []

    todos = generar_slots(fecha)
    ocupados = await get_slots_ocupados(fecha)
    disponibles = [s for s in todos if s not in ocupados]

    # Filtrar slots ocupados en Google Calendar
    if gcal_busy:
        from zoneinfo import ZoneInfo
        lima = ZoneInfo("America/Lima")
        filtered = []
        for slot in disponibles:
            slot_start = datetime.combine(fecha, slot).replace(tzinfo=lima)
            slot_end = slot_start + timedelta(minutes=INTERVALO_CITAS_MIN)
            busy = False
            for bs, be in gcal_busy:
                if bs.tzinfo:
                    bs = bs.astimezone(lima)
                    be = be.astimezone(lima)
                if slot_start < be and slot_end > bs:
                    busy = True
                    break
            if not busy:
                filtered.append(slot)
        disponibles = filtered

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


async def registrar_ejecucion(
    conversation_id: int,
    contact_name: str,
    canal: str,
    mensaje_usuario: str,
    respuesta_agente: str,
    tipo: str = "texto",
    cita_creada: bool = False,
    contact_phone: str = "",
):
    """Registra una ejecucion del agente (para el log del dashboard)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO ejecuciones (conversation_id, contact_name, contact_phone, canal, mensaje_usuario, respuesta_agente, tipo, cita_creada)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """, conversation_id, contact_name, contact_phone, canal, mensaje_usuario[:500], respuesta_agente[:500], tipo, cita_creada)


async def get_ejecuciones(limit: int = 50) -> list[dict]:
    """Retorna las ultimas ejecuciones del agente."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM ejecuciones
            ORDER BY created_at DESC
            LIMIT $1
        """, limit)
        return [dict(r) for r in rows]


async def get_conversaciones(limit: int = 50) -> list[dict]:
    """Retorna conversaciones agrupadas por conversation_id con resumen."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                conversation_id,
                MAX(contact_name) AS contact_name,
                MAX(contact_phone) AS contact_phone,
                MAX(canal) AS canal,
                COUNT(*) AS total_mensajes,
                BOOL_OR(cita_creada) AS cita_creada,
                MIN(created_at) AS primera_interaccion,
                MAX(created_at) AS ultima_interaccion,
                (SELECT mensaje_usuario FROM ejecuciones e2
                 WHERE e2.conversation_id = e.conversation_id
                 AND e2.tipo != 'seguimiento'
                 ORDER BY e2.created_at ASC LIMIT 1) AS primer_mensaje
            FROM ejecuciones e
            WHERE conversation_id IS NOT NULL
            GROUP BY conversation_id
            ORDER BY MAX(created_at) DESC
            LIMIT $1
        """, limit)
        return [dict(r) for r in rows]


async def get_hilo_conversacion(conversation_id: int) -> list[dict]:
    """Retorna todos los mensajes de una conversacion en orden cronologico."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM ejecuciones
            WHERE conversation_id = $1
            ORDER BY created_at ASC
        """, conversation_id)
        return [dict(r) for r in rows]


async def upsert_seguimiento(conversation_id: int, contact_name: str, cita_creada: bool = False):
    """Actualiza o crea el tracking de seguimiento para una conversacion."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if cita_creada:
            # Cerrar seguimiento si se creo cita
            await conn.execute("""
                UPDATE seguimientos SET cita_creada = TRUE, cerrado = TRUE
                WHERE conversation_id = $1 AND NOT cerrado
            """, conversation_id)
            return

        # Upsert: actualizar timestamp o crear nuevo
        existing = await conn.fetchrow("""
            SELECT id FROM seguimientos
            WHERE conversation_id = $1 AND NOT cerrado
        """, conversation_id)

        if existing:
            await conn.execute("""
                UPDATE seguimientos SET ultimo_mensaje_at = NOW(), contact_name = $2
                WHERE id = $3
            """, conversation_id, contact_name, existing["id"])
        else:
            await conn.execute("""
                INSERT INTO seguimientos (conversation_id, contact_name, ultimo_mensaje_at)
                VALUES ($1, $2, NOW())
            """, conversation_id, contact_name)


async def get_seguimientos_pendientes() -> list[dict]:
    """
    Retorna conversaciones que necesitan seguimiento.
    - Seguimiento 1: >= 2 horas sin respuesta, aun no enviado
    - Seguimiento 2: >= 8 horas sin respuesta, seguimiento 1 ya enviado
    - Dentro de ventana 24h de Meta
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM seguimientos
            WHERE NOT cerrado
              AND NOT cita_creada
              AND ultimo_mensaje_at > NOW() - INTERVAL '24 hours'
              AND (
                  (seguimiento_num = 0 AND ultimo_mensaje_at < NOW() - INTERVAL '2 hours')
                  OR
                  (seguimiento_num = 1 AND seguimiento1_at < NOW() - INTERVAL '6 hours')
              )
            ORDER BY ultimo_mensaje_at ASC
        """)
        return [dict(r) for r in rows]


async def marcar_seguimiento_enviado(seguimiento_id: int, num: int):
    """Marca un seguimiento como enviado."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if num == 1:
            await conn.execute("""
                UPDATE seguimientos SET seguimiento_num = 1, seguimiento1_at = NOW()
                WHERE id = $1
            """, seguimiento_id)
        elif num == 2:
            await conn.execute("""
                UPDATE seguimientos SET seguimiento_num = 2, seguimiento2_at = NOW(), cerrado = TRUE
                WHERE id = $1
            """, seguimiento_id)


async def cerrar_seguimientos_expirados():
    """Cierra seguimientos que pasaron las 24h de Meta."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE seguimientos SET cerrado = TRUE
            WHERE NOT cerrado AND ultimo_mensaje_at < NOW() - INTERVAL '24 hours'
        """)


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


# ── Google Calendar tokens ──

async def save_gcal_token(token_data: dict, email: str = ""):
    """Guarda o actualiza el token de Google Calendar (uno solo, el primario)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow("SELECT id FROM gcal_tokens WHERE is_primary = TRUE")
        td = json.dumps(token_data)
        if existing:
            await conn.execute(
                "UPDATE gcal_tokens SET token_data = $1, email = $2, updated_at = NOW() WHERE id = $3",
                td, email, existing["id"],
            )
        else:
            await conn.execute(
                "INSERT INTO gcal_tokens (token_data, email, is_primary) VALUES ($1, $2, TRUE)",
                td, email,
            )


async def get_gcal_token() -> dict | None:
    """Retorna el token primario de Google Calendar."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT token_data, email FROM gcal_tokens WHERE is_primary = TRUE")
        if row:
            data = json.loads(row["token_data"]) if isinstance(row["token_data"], str) else row["token_data"]
            data["_email"] = row["email"] or ""
            return data
    return None


async def delete_gcal_token():
    """Elimina la conexion con Google Calendar."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM gcal_tokens WHERE is_primary = TRUE")


async def update_cita_gcal_id(cita_id: int, gcal_event_id: str):
    """Guarda el ID del evento GCal en la cita."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("UPDATE citas SET gcal_event_id = $1 WHERE id = $2", gcal_event_id, cita_id)


async def get_cita(cita_id: int) -> dict | None:
    """Retorna una cita por ID."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM citas WHERE id = $1", cita_id)
        return dict(row) if row else None
