"""Gestion de citas — base de datos PostgreSQL."""

import json
import logging
from datetime import date, time, datetime, timedelta
from typing import Optional
import asyncpg
from config import DATABASE_URL, HORARIO_INICIO, HORARIO_FIN, HORARIO_FIN_SABADO, INTERVALO_CITAS_MIN, SLOTS_VISIBLES, TURNO_MANANA, TURNO_TARDE
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

            -- Agregar columna tipo_paciente si no existe (migracion v32)
            DO $$ BEGIN
                ALTER TABLE citas ADD COLUMN IF NOT EXISTS tipo_paciente TEXT;
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

            -- Columna interesado: solo enviar seguimiento si paciente mostró interés
            DO $$ BEGIN
                ALTER TABLE seguimientos ADD COLUMN IF NOT EXISTS interesado BOOLEAN DEFAULT FALSE;
            EXCEPTION WHEN OTHERS THEN NULL;
            END $$;

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

            -- Turnos bloqueados (equipo bloquea manana o tarde de un dia especifico)
            CREATE TABLE IF NOT EXISTS turnos_bloqueados (
                id SERIAL PRIMARY KEY,
                fecha DATE NOT NULL,
                turno TEXT NOT NULL CHECK (turno IN ('manana', 'tarde')),
                motivo TEXT DEFAULT 'sin cupos',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(fecha, turno)
            );

            -- Alertas (handoff a supervisor + citas agendadas)
            CREATE TABLE IF NOT EXISTS alertas (
                id SERIAL PRIMARY KEY,
                tipo TEXT NOT NULL,
                conversation_id INTEGER,
                contact_name TEXT,
                contact_phone TEXT,
                detalle TEXT,
                leida BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_alertas_created ON alertas(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_alertas_leida ON alertas(leida);
        """)
        log.info("DB inicializada: tablas citas y ejecuciones listas")


async def crear_cita(cita: Cita) -> int | None:
    """Crea una cita y retorna el ID. Previene duplicados por conversation_id Y por fecha+hora."""
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

        # Verificar turno bloqueado antes de crear
        turno = "manana" if cita.hora.hour < 12 else "tarde"
        turno_bloq = await conn.fetchval(
            "SELECT 1 FROM turnos_bloqueados WHERE fecha = $1 AND turno = $2",
            cita.fecha, turno,
        )
        if turno_bloq:
            log.warning(f"TURNO BLOQUEADO: {cita.fecha} {turno} — no se crea cita")
            return None

        # Prevenir double-booking con lock advisory para evitar race conditions
        # Lock basado en fecha+hora (hash único por slot)
        lock_key = int(cita.fecha.toordinal()) * 10000 + cita.hora.hour * 100 + cita.hora.minute
        await conn.execute("SELECT pg_advisory_xact_lock($1)", lock_key)

        slot_taken = await conn.fetchval("""
            SELECT id FROM citas
            WHERE fecha = $1 AND hora = $2 AND estado NOT IN ('cancelada')
            LIMIT 1
        """, cita.fecha, cita.hora)
        if slot_taken:
            log.warning(f"SLOT OCUPADO: {cita.fecha} {cita.hora} ya tiene cita #{slot_taken} — no se crea duplicado")
            return None

        row = await conn.fetchrow("""
            INSERT INTO citas (nombre_paciente, telefono, fecha, hora, motivo, canal, estado, conversation_id, contact_id, tipo_paciente)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
        """, cita.nombre_paciente, cita.telefono, cita.fecha, cita.hora,
            cita.motivo, cita.canal.value, cita.estado.value,
            cita.conversation_id, cita.contact_id, cita.tipo_paciente)
        log.info(f"Cita creada #{row['id']}: {cita.nombre_paciente} - {cita.fecha} {cita.hora}")
        return row["id"]


async def buscar_citas_por_nombre(nombre: str) -> list[dict]:
    """Busca citas activas (futuras) por nombre de paciente (búsqueda parcial, case-insensitive)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, nombre_paciente, telefono, fecha, hora, motivo, estado, conversation_id
            FROM citas
            WHERE LOWER(nombre_paciente) LIKE '%' || LOWER($1) || '%'
            AND estado NOT IN ('cancelada', 'bloqueado')
            AND fecha >= CURRENT_DATE
            ORDER BY fecha, hora
            LIMIT 5
        """, nombre)
        return [dict(r) for r in rows]


async def buscar_citas_por_telefono(telefono: str) -> list[dict]:
    """Busca citas activas (futuras) por teléfono."""
    pool = await get_pool()
    # Limpiar: quedarnos solo con dígitos y los últimos 9
    tel_limpio = ''.join(c for c in telefono if c.isdigit())[-9:]
    if len(tel_limpio) < 7:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, nombre_paciente, telefono, fecha, hora, motivo, estado, conversation_id
            FROM citas
            WHERE telefono LIKE '%' || $1
            AND estado NOT IN ('cancelada', 'bloqueado')
            AND fecha >= CURRENT_DATE
            ORDER BY fecha, hora
            LIMIT 5
        """, tel_limpio)
        return [dict(r) for r in rows]


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


async def contar_citas_dia(fecha: date) -> int:
    """Retorna el número de citas activas (no canceladas) de un día."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT COUNT(*) FROM citas WHERE fecha = $1 AND estado NOT IN ('cancelada')",
            fecha,
        )


def generar_slots(fecha: date) -> list[time]:
    """Genera slots por turnos: mañana 8:30-11:00 (Lun-Sab), tarde 14:00-15:40 (solo Lun-Vie)."""
    slots = []
    # Turno mañana (Lun-Sab)
    inicio_m = datetime.strptime(TURNO_MANANA[0], "%H:%M")
    fin_m = datetime.strptime(TURNO_MANANA[1], "%H:%M")
    current = inicio_m
    while current <= fin_m:
        slots.append(current.time())
        current += timedelta(minutes=INTERVALO_CITAS_MIN)
    # Turno tarde (solo Lun-Vie, no Sab)
    if fecha.weekday() < 5:
        inicio_t = datetime.strptime(TURNO_TARDE[0], "%H:%M")
        fin_t = datetime.strptime(TURNO_TARDE[1], "%H:%M")
        current = inicio_t
        while current <= fin_t:
            slots.append(current.time())
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


async def bloquear_turno(fecha: date, turno: str, motivo: str = "sin cupos"):
    """Bloquea un turno (manana/tarde) de un dia especifico."""
    if turno not in ("manana", "tarde"):
        raise ValueError(f"Turno invalido: {turno}. Usar 'manana' o 'tarde'.")
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO turnos_bloqueados (fecha, turno, motivo) VALUES ($1, $2, $3)
            ON CONFLICT (fecha, turno) DO UPDATE SET motivo = $3
        """, fecha, turno, motivo)
    log.info(f"Turno bloqueado: {fecha} {turno} ({motivo})")


async def desbloquear_turno(fecha: date, turno: str):
    """Desbloquea un turno de un dia."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM turnos_bloqueados WHERE fecha = $1 AND turno = $2",
            fecha, turno,
        )
    log.info(f"Turno desbloqueado: {fecha} {turno}")


async def get_turnos_bloqueados(fecha: date | None = None) -> list[dict]:
    """Lista turnos bloqueados. Si fecha es None, retorna todos los futuros."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if fecha:
            rows = await conn.fetch(
                "SELECT fecha, turno, motivo FROM turnos_bloqueados WHERE fecha = $1 ORDER BY turno",
                fecha,
            )
        else:
            rows = await conn.fetch(
                "SELECT fecha, turno, motivo FROM turnos_bloqueados WHERE fecha >= CURRENT_DATE ORDER BY fecha, turno"
            )
        return [{"fecha": r["fecha"].isoformat(), "turno": r["turno"], "motivo": r["motivo"]} for r in rows]


async def is_turno_bloqueado(fecha: date, turno: str) -> bool:
    """Verifica si un turno esta bloqueado."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchval(
            "SELECT 1 FROM turnos_bloqueados WHERE fecha = $1 AND turno = $2",
            fecha, turno,
        )
        return row is not None


async def detectar_y_bloquear_turno_lleno(fecha: date, gcal_busy: list | None = None):
    """
    Detecta si un turno (mañana/tarde) está lleno combinando DB + GCal
    y lo auto-bloquea si no quedan slots. Retorna los turnos bloqueados.
    """
    if await is_dia_bloqueado(fecha):
        return []

    todos = generar_slots(fecha)
    ocupados_db = await get_slots_ocupados(fecha)
    disponibles = [s for s in todos if s not in ocupados_db]

    # Aplicar filtro GCal
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

    # Separar por turno
    manana_disp = [s for s in disponibles if s.hour < 12]
    tarde_disp = [s for s in disponibles if s.hour >= 12]

    turnos_auto = []

    # Solo auto-bloquear si el turno tiene slots definidos pero 0 disponibles
    todos_manana = [s for s in todos if s.hour < 12]
    todos_tarde = [s for s in todos if s.hour >= 12]

    if todos_manana and not manana_disp and not await is_turno_bloqueado(fecha, "manana"):
        await bloquear_turno(fecha, "manana", "auto: sin cupos (GCal+DB)")
        log.warning(f"AUTO-BLOQUEO turno mañana {fecha}: 0/{len(todos_manana)} slots disponibles")
        turnos_auto.append("manana")

    if todos_tarde and not tarde_disp and not await is_turno_bloqueado(fecha, "tarde"):
        await bloquear_turno(fecha, "tarde", "auto: sin cupos (GCal+DB)")
        log.warning(f"AUTO-BLOQUEO turno tarde {fecha}: 0/{len(todos_tarde)} slots disponibles")
        turnos_auto.append("tarde")

    return turnos_auto


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

    # Filtrar turnos bloqueados (manana < 12:00, tarde >= 12:00)
    manana_bloq = await is_turno_bloqueado(fecha, "manana")
    tarde_bloq = await is_turno_bloqueado(fecha, "tarde")
    if manana_bloq:
        disponibles = [s for s in disponibles if s.hour >= 12]
    if tarde_bloq:
        disponibles = [s for s in disponibles if s.hour < 12]

    # Si es hoy, filtrar slots que ya pasaron
    from zoneinfo import ZoneInfo
    ahora = datetime.now(ZoneInfo("America/Lima"))
    if fecha == ahora.date():
        hora_actual = ahora.time()
        disponibles = [s for s in disponibles if s > hora_actual]

    # Filtrar slots ocupados en Google Calendar
    # gcal_busy=None means GCal error (token expired, etc) — log warning
    # gcal_busy=[] means no busy slots — normal
    if gcal_busy is None:
        log.warning(f"GCal no disponible para {fecha} — slots basados solo en DB (pueden faltar citas externas)")
    elif gcal_busy:
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
        log.info(f"GCal filtro {fecha}: {len(disponibles)} -> {len(filtered)} slots (eliminó {len(disponibles)-len(filtered)})")
        disponibles = filtered

    if not disponibles:
        return []

    # Separar mañana y tarde para garantizar representación de ambos turnos
    manana = [s for s in disponibles if s.hour < 12]
    tarde = [s for s in disponibles if s.hour >= 12]

    if len(disponibles) <= SLOTS_VISIBLES:
        seleccion = disponibles
    elif manana and tarde:
        # Garantizar al menos 1 slot de cada turno
        # 2 de mañana (inicio y medio) + 1 de tarde (inicio)
        m_step = max(1, len(manana) // 2)
        seleccion = [manana[0], manana[min(m_step, len(manana)-1)], tarde[0]]
    elif manana:
        step = max(1, len(manana) // SLOTS_VISIBLES)
        seleccion = [manana[i * step] for i in range(min(SLOTS_VISIBLES, len(manana)))]
    else:
        step = max(1, len(tarde) // SLOTS_VISIBLES)
        seleccion = [tarde[i * step] for i in range(min(SLOTS_VISIBLES, len(tarde)))]

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


async def get_ultima_ejecucion(conversation_id: int) -> dict | None:
    """Retorna la última ejecución (no seguimiento) de una conversación."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT * FROM ejecuciones
            WHERE conversation_id = $1 AND tipo != 'seguimiento'
            ORDER BY created_at DESC LIMIT 1
        """, conversation_id)
        return dict(row) if row else None


async def upsert_seguimiento(conversation_id: int, contact_name: str, cita_creada: bool = False, interesado: bool = False):
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
            SELECT id, interesado FROM seguimientos
            WHERE conversation_id = $1 AND NOT cerrado
        """, conversation_id)

        if existing:
            # Si ya estaba marcado como interesado, no quitarlo
            new_interesado = existing["interesado"] or interesado
            await conn.execute("""
                UPDATE seguimientos SET ultimo_mensaje_at = NOW(), contact_name = $1, interesado = $3
                WHERE id = $2
            """, contact_name, existing["id"], new_interesado)
        else:
            await conn.execute("""
                INSERT INTO seguimientos (conversation_id, contact_name, ultimo_mensaje_at, interesado)
                VALUES ($1, $2, NOW(), $3)
            """, conversation_id, contact_name, interesado)


async def get_seguimientos_pendientes() -> list[dict]:
    """
    Retorna conversaciones que necesitan seguimiento.
    - Solo 1 seguimiento: >= 2 horas sin respuesta, aun no enviado
    - Dentro de ventana 24h de Meta
    - Solo si el paciente mostró interés (interesado = TRUE)
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM seguimientos
            WHERE NOT cerrado
              AND NOT cita_creada
              AND interesado = TRUE
              AND seguimiento_num = 0
              AND ultimo_mensaje_at > NOW() - INTERVAL '24 hours'
              AND ultimo_mensaje_at < NOW() - INTERVAL '2 hours'
            ORDER BY ultimo_mensaje_at ASC
        """)
        return [dict(r) for r in rows]


async def marcar_seguimiento_enviado(seguimiento_id: int, num: int):
    """Marca un seguimiento como enviado y cierra (solo 1 seguimiento)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE seguimientos SET seguimiento_num = 1, seguimiento1_at = NOW(), cerrado = TRUE
            WHERE id = $1
        """, seguimiento_id)


async def cerrar_seguimiento(conversation_id: int):
    """Cierra seguimiento de una conversación (handoff o cita creada)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE seguimientos SET cerrado = TRUE
            WHERE conversation_id = $1 AND NOT cerrado
        """, conversation_id)


async def cerrar_seguimientos_expirados():
    """Cierra seguimientos que pasaron las 24h de Meta."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE seguimientos SET cerrado = TRUE
            WHERE NOT cerrado AND ultimo_mensaje_at < NOW() - INTERVAL '24 hours'
        """)


async def registrar_alerta(tipo: str, conversation_id: int = None, contact_name: str = "", contact_phone: str = "", detalle: str = ""):
    """Registra una alerta (supervisor o cita_agendada)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO alertas (tipo, conversation_id, contact_name, contact_phone, detalle)
            VALUES ($1, $2, $3, $4, $5)
        """, tipo, conversation_id, contact_name, contact_phone, detalle)


async def get_alertas(limit: int = 50, solo_no_leidas: bool = False) -> list[dict]:
    """Retorna alertas recientes."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        where = "WHERE NOT leida" if solo_no_leidas else ""
        rows = await conn.fetch(f"""
            SELECT * FROM alertas {where}
            ORDER BY created_at DESC LIMIT $1
        """, limit)
        return [dict(r) for r in rows]


async def marcar_alertas_leidas(ids: list[int] = None):
    """Marca alertas como leidas. Si ids=None, marca todas."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if ids:
            await conn.execute("UPDATE alertas SET leida = TRUE WHERE id = ANY($1)", ids)
        else:
            await conn.execute("UPDATE alertas SET leida = TRUE WHERE NOT leida")


async def contar_alertas_no_leidas() -> int:
    """Cuenta alertas no leidas."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT COUNT(*) FROM alertas WHERE NOT leida")


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
