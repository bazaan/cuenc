# Manual de Operacion y Soporte — Agente IA Clinica Respira Vida

## Indice

1. [Que es y como funciona](#1-que-es-y-como-funciona)
2. [Arquitectura del sistema](#2-arquitectura-del-sistema)
3. [Accesos y credenciales](#3-accesos-y-credenciales)
4. [Operacion diaria — Dashboard](#4-operacion-diaria--dashboard)
5. [Conectarse al servidor (VPS)](#5-conectarse-al-servidor-vps)
6. [Ver logs del bot en tiempo real](#6-ver-logs-del-bot-en-tiempo-real)
7. [Encender / apagar la IA](#7-encender--apagar-la-ia)
8. [Que hacer si el bot no responde](#8-que-hacer-si-el-bot-no-responde)
9. [Modificar el prompt (lo que dice el bot)](#9-modificar-el-prompt-lo-que-dice-el-bot)
10. [Modificar precios](#10-modificar-precios)
11. [Modificar horarios de atencion](#11-modificar-horarios-de-atencion)
12. [Hacer deploy (subir cambios al servidor)](#12-hacer-deploy-subir-cambios-al-servidor)
13. [Enviar mensajes manuales via API](#13-enviar-mensajes-manuales-via-api)
14. [Estructura de archivos](#14-estructura-de-archivos)
15. [Base de datos — consultar citas](#15-base-de-datos--consultar-citas)
16. [Problemas comunes y soluciones](#16-problemas-comunes-y-soluciones)

---

## 1. Que es y como funciona

El agente es un **bot de WhatsApp** que atiende pacientes de la Clinica Respira Vida automaticamente. Cuando un paciente escribe por WhatsApp:

1. **Chatwoot** (plataforma de chat) recibe el mensaje
2. Chatwoot envia un **webhook** al bot (notificacion HTTP)
3. El bot procesa el mensaje:
   - Si es texto: lo analiza directamente
   - Si es audio: lo transcribe con **Whisper** (OpenAI) a texto
   - Si es imagen: la describe con **GPT-4o Vision**
4. El bot genera una respuesta usando **GPT-5.4-mini** (OpenAI)
5. Envia la respuesta de vuelta al paciente via Chatwoot

El bot puede:
- Dar informacion de precios, horarios, direccion
- Agendar citas (las guarda en base de datos)
- Derivar a supervisor cuando es necesario (controles, quejas, casos especiales)
- Detectar restricciones medicas (TBC, oncologico, embarazadas)

---

## 2. Arquitectura del sistema

```
Paciente (WhatsApp)
       |
       v
  Chatwoot (chats.alef.company)
       |  webhook
       v
  Bot FastAPI (puerto 8090)
       |
       +-- OpenAI API (GPT-5.4-mini, Whisper)
       +-- PostgreSQL (citas, alertas)
       +-- Redis (estado conversaciones)
       |
       v
  Dashboard web (dashboard.respiravida.online)
```

**Donde corre todo:**
- **VPS (servidor):** IP `173.249.59.135` — Linux con Docker
- Dentro del servidor hay 3 containers Docker:
  - `docc_agent_agent` — el bot (FastAPI, Python)
  - `docc_agent_docc_redis` — Redis (memoria temporal de conversaciones)
  - `docc_agent_docc_pg` — PostgreSQL (base de datos de citas)

---

## 3. Accesos y credenciales

### Dashboard (panel de citas)
- URL: https://dashboard.respiravida.online/dashboard
- Usuarios:
  - `admin` / `respiravida2024` (administrador)
  - `nanenafriend13@gmail.com` / `Huracan126` (staff)
  - `juliozumaeta@alef.com` / `Alef@2026` (administrador)

### Chatwoot (chat con pacientes)
- URL: https://chats.alef.company
- Account ID: **16**
- API Token: `xBsW4FE3FCZdZbgXgdjrHfUA`

### Servidor VPS
- IP: `173.249.59.135`
- Usuario: `root`
- Acceso: por SSH (necesitas la clave SSH configurada)

### OpenAI
- API Key: configurada en el container como variable de entorno
- Modelo: `gpt-5.4-mini`

---

## 4. Operacion diaria -- Dashboard

### Acceder
1. Ir a https://dashboard.respiravida.online/dashboard
2. Iniciar sesion con tu usuario

### Funciones principales

**Ver citas del dia:**
- El calendario muestra las citas en dos turnos: manana (8:30-11:00) y tarde (14:00-15:40)
- Colores: verde = confirmada, amarillo = pendiente, rojo = cancelada, gris = bloqueado

**Encender/apagar la IA:**
- En la barra superior hay un switch "IA Activa / Pausada"
- Si la apagas, el bot deja de responder y los pacientes solo veran el menu automatico

**Bloquear horarios:**
- Click en "Ocupado Rapido" en el calendario
- Click en las ranuras de 10 minutos que quieras bloquear
- Los slots bloqueados no se ofrecen a pacientes

**Ver alertas:**
- Seccion de alertas muestra: nuevas citas, handoffs, recuperaciones del watchdog
- Se actualizan cada 15 segundos

**Ir al chat del paciente:**
- Cada cita tiene un boton "CRM" que abre la conversacion en Chatwoot

---

## 5. Conectarse al servidor (VPS)

### Desde Mac/Linux (Terminal)
```bash
ssh root@173.249.59.135
```

### Desde Windows
1. Descargar [PuTTY](https://www.putty.org/) o usar Windows Terminal
2. Conectar a: `root@173.249.59.135`

### Una vez dentro del servidor

Ver containers corriendo:
```bash
docker ps
```

Deberias ver 3 containers con nombres que incluyen `docc_agent`.

---

## 6. Ver logs del bot en tiempo real

Conectate al servidor y ejecuta:

```bash
# Ver las ultimas 50 lineas de log
docker logs $(docker ps -q -f name=docc_agent_agent.1) --tail 50

# Ver logs EN VIVO (Ctrl+C para salir)
docker logs $(docker ps -q -f name=docc_agent_agent.1) --tail 20 -f
```

**Que buscar en los logs:**

| Log | Significado |
|-----|-------------|
| `HTTP Request: POST ...chat/completions` | El bot llamo a OpenAI (funciona) |
| `Audio transcrito` | Whisper transcribio un audio |
| `CRITICAL...SIN CREDITO` | OpenAI sin saldo — bot silenciado |
| `Error OpenAI API` | Fallo temporal de OpenAI |
| `watchdog_recovery` | El watchdog detecto un mensaje sin respuesta |
| `CITA DUPLICADA ignorada` | El bot evito crear cita repetida |

---

## 7. Encender / apagar la IA

### Opcion 1: Desde el Dashboard (recomendado)
- Toggle en la barra superior del dashboard

### Opcion 2: Desde el servidor
```bash
# Ver estado actual
docker service inspect docc_agent_agent --format '{{range .Spec.TaskTemplate.ContainerSpec.Env}}{{println .}}{{end}}' | grep AI_ENABLED

# Encender
docker service update --env-add AI_ENABLED=true docc_agent_agent

# Apagar
docker service update --env-add AI_ENABLED=false docc_agent_agent
```

**IMPORTANTE:** Cada vez que se hace deploy de una nueva version, AI_ENABLED vuelve a `false`. Hay que activarlo despues.

---

## 8. Que hacer si el bot no responde

### Paso 1: Verificar si la IA esta encendida
- Dashboard > toggle de IA debe decir "Activa"
- O verificar en servidor: `AI_ENABLED=true`

### Paso 2: Ver logs
```bash
docker logs $(docker ps -q -f name=docc_agent_agent.1) --tail 20
```

### Paso 3: Identificar el problema

| Sintoma | Causa probable | Solucion |
|---------|----------------|----------|
| Bot no responde nada | IA apagada | Encender IA |
| Bot no responde nada | Sin credito OpenAI | Recargar credito en platform.openai.com |
| "En que puedo ayudarle?" repetido | Error de API (pero no quota) | Ver logs, puede ser temporal |
| Solo sale menu automatico | Webhook no llega | Verificar configuracion webhook en Chatwoot |
| Responde lento (>30s) | OpenAI lento | Normal en horas pico, el watchdog cubre |

### Paso 4: Si es por credito OpenAI
1. Ir a https://platform.openai.com/settings/organization/billing
2. Agregar credito ($5-15 USD)
3. El bot se recupera solo (no necesita restart)

### Paso 5: Reiniciar el bot (ultimo recurso)
```bash
docker service update --force docc_agent_agent
```

---

## 9. Modificar el prompt (lo que dice el bot)

El prompt esta en el archivo `ai_engine.py`, variable `SYSTEM_PROMPT` (lineas 14-152).

### Para editarlo:

**Opcion 1: Desde tu PC (recomendado)**
1. Tener el archivo `ai_engine.py` en tu computadora
2. Abrirlo con cualquier editor de texto (VS Code, Notepad++, etc.)
3. Modificar el texto dentro de `SYSTEM_PROMPT = """..."""`
4. Hacer deploy (ver seccion 12)

**Opcion 2: Directo en el servidor**
```bash
ssh root@173.249.59.135
nano /root/doc-c-agent/ai_engine.py
# Editar, guardar con Ctrl+O, salir con Ctrl+X
# Luego hacer deploy (seccion 12)
```

### Que se puede cambiar:

- **Personalidad:** linea 16 — tono, emojis, formato
- **Reglas:** lineas 18-41 — comportamiento del bot
- **Flujo de conversacion:** lineas 43-84 — pasos que sigue
- **Datos clinica:** lineas 103-112 — direccion, precios, horarios, servicios
- **Restricciones medicas:** lineas 86-91 — que NO agendar
- **Objeciones:** lineas 93-98 — como responder a dudas
- **Ejemplos:** lineas 120-151 — guian al modelo sobre como responder

### Tips para editar el prompt:
- Ser especifico: "Di exactamente esto" funciona mejor que "se amable"
- Usar ejemplos: el bot imita los ejemplos que le des
- Probar despues de cada cambio — un cambio pequeno puede afectar mucho
- Si algo no funciona, el bot probablemente no entiende la instruccion. Reformularla.

---

## 10. Modificar precios

Los precios estan en DOS lugares:

### 1. Texto exacto que recibe el paciente
Archivo: `ai_engine.py`, variable `PRECIO_MESSAGE` (lineas 155-160):
```python
PRECIO_MESSAGE = """Costo de consulta : S70.00
Examenes de laboratorio: s/.100.00 a s/.200.00

=> Aceptamos efectivo, tarjetas y transferencias.

Tenga en cuenta que si el paciente presenta enfermedad respiratoria..."""
```
Modificar este texto directamente. Es EXACTAMENTE lo que recibe el paciente.

### 2. Referencia en el prompt para el bot
Archivo: `ai_engine.py`, dentro de `SYSTEM_PROMPT`, linea 109:
```
Consulta S/70 (se paga despues). Vacuna influenza S/80. Panel alergias S/170...
```

**IMPORTANTE:** Cambiar en AMBOS lugares para mantener consistencia.

---

## 11. Modificar horarios de atencion

Archivo: `config.py`

```python
TURNO_MANANA = ("08:30", "11:00")   # Lunes a Sabado
TURNO_TARDE = ("14:00", "15:40")    # Solo Lunes a Viernes
INTERVALO_CITAS_MIN = 10            # Cita cada 10 minutos
SLOTS_VISIBLES = 3                  # Mostrar 3 horarios al paciente
CITAS_DIA_LLENO = 12                # Maximo citas por dia antes de marcar "lleno"
```

Tambien actualizar el prompt en `ai_engine.py` linea 108:
```
Lunes-Viernes manana y tarde. Sabados solo manana. Domingos NO.
```

---

## 12. Hacer deploy (subir cambios al servidor)

### Desde tu PC (Mac/Linux):

```bash
# 1. Subir archivos modificados al servidor
scp ai_engine.py root@173.249.59.135:/root/doc-c-agent/
scp config.py root@173.249.59.135:/root/doc-c-agent/
# (subir solo los archivos que cambiaste)

# 2. Conectar al servidor
ssh root@173.249.59.135

# 3. Construir nueva version (cambiar vN por el numero de version)
cd /root/doc-c-agent
docker build -t doc-c-agent:v63 .

# 4. Desplegar + activar IA
docker service update --image doc-c-agent:v63 --env-add AI_ENABLED=true docc_agent_agent

# 5. Verificar que arranco bien
docker logs $(docker ps -q -f name=docc_agent_agent.1) --tail 10
```

### Desde Windows:
1. Usar WinSCP para subir archivos a `/root/doc-c-agent/`
2. Usar PuTTY para conectar y ejecutar los comandos del paso 3 en adelante

### Cual version usar?
Ver la version actual:
```bash
docker service inspect docc_agent_agent --format '{{.Spec.TaskTemplate.ContainerSpec.Image}}'
```
Ejemplo: si dice `doc-c-agent:v62`, la siguiente es `v63`.

---

## 13. Enviar mensajes manuales via API

Para enviar un mensaje a un paciente desde la terminal (sin abrir Chatwoot):

```bash
curl -X POST "https://chats.alef.company/api/v1/accounts/16/conversations/CONV_ID/messages" \
  -H "api_access_token: xBsW4FE3FCZdZbgXgdjrHfUA" \
  -H "Content-Type: application/json" \
  -d '{"content": "Tu mensaje aqui", "message_type": "outgoing"}'
```

Reemplazar `CONV_ID` con el numero de conversacion (se ve en la URL de Chatwoot).

### Buscar un paciente por telefono:
```bash
curl -s "https://chats.alef.company/api/v1/accounts/16/contacts/search?q=TELEFONO" \
  -H "api_access_token: xBsW4FE3FCZdZbgXgdjrHfUA"
```

---

## 14. Estructura de archivos

```
doc-c-agent/
  ai_engine.py        # Prompt del bot + generacion de respuestas (OpenAI)
  main.py             # Servidor principal: webhook, API, watchdog
  config.py           # Configuracion: horarios, URLs, tokens, credenciales
  models.py           # Modelos de datos (Cita, ConversationState)
  appointments.py     # Logica de citas (crear, buscar, cancelar)
  chatwoot_client.py  # Comunicacion con Chatwoot (enviar mensajes, labels)
  conversation.py     # Estado de conversacion en Redis
  transcriber.py      # Transcripcion de audios (Whisper) e imagenes (Vision)
  gcal.py             # Integracion Google Calendar (pendiente credenciales)
  auth.py             # Autenticacion JWT para el dashboard
  requirements.txt    # Dependencias Python
  Dockerfile          # Como construir el container
  static/
    dashboard.html    # Panel principal de citas y calendario
    login.html        # Pagina de login
    panel.html        # Panel alternativo
```

### Que archivo editar para cada cosa:

| Quiero cambiar... | Archivo |
|--------------------|---------|
| Lo que dice el bot | `ai_engine.py` (SYSTEM_PROMPT) |
| Precios | `ai_engine.py` (PRECIO_MESSAGE + SYSTEM_PROMPT) |
| Horarios | `config.py` (TURNO_MANANA, TURNO_TARDE) |
| Datos de la clinica | `ai_engine.py` (SYSTEM_PROMPT seccion DATOS CLINICA) |
| Mensaje de bienvenida | `main.py` (buscar "greeting" o "menu de opciones") |
| Credenciales | `config.py` o variables de entorno en Docker |
| Aspecto del dashboard | `static/dashboard.html` |
| Usuarios del dashboard | `auth.py` o endpoint de API |

---

## 15. Base de datos -- consultar citas

### Desde el servidor:

```bash
# Entrar a PostgreSQL
docker exec -it $(docker ps -q -f name=docc_agent_docc_pg) psql -U docc -d docc_agent

# Ver citas de hoy
SELECT nombre_paciente, hora, estado FROM citas WHERE fecha = CURRENT_DATE ORDER BY hora;

# Ver citas de una fecha
SELECT * FROM citas WHERE fecha = '2026-06-05' ORDER BY hora;

# Buscar por nombre
SELECT * FROM citas WHERE nombre_paciente ILIKE '%hilda%';

# Buscar por telefono
SELECT * FROM citas WHERE telefono LIKE '%962594206%';

# Contar citas por dia
SELECT fecha, COUNT(*) FROM citas WHERE estado != 'cancelada' GROUP BY fecha ORDER BY fecha;

# Salir
\q
```

### Borrar memoria de conversacion (Redis):
```bash
# Borrar memoria de un paciente especifico (por telefono)
docker exec $(docker ps -q -f name=docc_agent_docc_redis) redis-cli DEL "conv:TELEFONO"

# Ver todas las conversaciones activas
docker exec $(docker ps -q -f name=docc_agent_docc_redis) redis-cli KEYS "conv:*"
```

---

## 16. Problemas comunes y soluciones

### El bot responde pero dice cosas raras
- **Causa:** El prompt necesita ajuste
- **Solucion:** Editar SYSTEM_PROMPT en `ai_engine.py`, agregar regla o ejemplo

### El bot no responde nada (silencio total)
1. Verificar IA encendida (dashboard toggle)
2. Ver logs: `docker logs ... --tail 20`
3. Si dice "SIN CREDITO": recargar en platform.openai.com
4. Si no hay logs recientes: `docker service update --force docc_agent_agent`

### Un paciente recibio respuesta repetitiva / mala
- Borrar su memoria en Redis para "resetear" la conversacion:
```bash
docker exec $(docker ps -q -f name=docc_agent_docc_redis) redis-cli DEL "conv:TELEFONO"
```

### El dashboard no carga
- Verificar que el container este corriendo: `docker ps`
- Si no esta: `docker service update --force docc_agent_agent`
- Si esta pero no carga: verificar DNS de dashboard.respiravida.online

### Se agoto el credito de OpenAI
1. El bot se silencia automaticamente (v62+, no marea al paciente)
2. Ir a https://platform.openai.com/settings/organization/billing
3. Agregar credito ($5-15 USD recomendado)
4. El bot se recupera automaticamente, no necesita reinicio

### Quiero que el bot deje de atender temporalmente
- Apagar desde dashboard (toggle IA)
- O desde servidor: `docker service update --env-add AI_ENABLED=false docc_agent_agent`
- Los pacientes veran solo el menu automatico, no se genera respuesta IA

### Un paciente fue derivado a supervisor por error
1. En Chatwoot, ir a la conversacion
2. Quitar label "supervisor" o "pasar_supervisor"
3. El bot retomara automaticamente

### Quiero cambiar la API key de OpenAI
```bash
docker service update --env-add OPENAI_API_KEY=sk-proj-NUEVA_KEY docc_agent_agent
```
No necesita rebuild, se reinicia automaticamente.

---

## Glosario

| Termino | Que es |
|---------|--------|
| **VPS** | Servidor virtual en la nube donde corre todo |
| **Docker** | Tecnologia que empaqueta la app en "containers" aislados |
| **Container** | Una caja que tiene la app corriendo dentro |
| **Deploy** | Subir una nueva version del bot al servidor |
| **Webhook** | Notificacion automatica que Chatwoot envia al bot |
| **Chatwoot** | Plataforma que conecta WhatsApp con el bot |
| **Redis** | Memoria rapida donde el bot guarda el estado de cada conversacion |
| **PostgreSQL** | Base de datos donde se guardan las citas |
| **Prompt** | Las instrucciones que el bot sigue para responder |
| **Watchdog** | Sistema que detecta mensajes sin respuesta y los procesa |
| **Handoff** | Cuando el bot pasa la conversacion a un humano |
| **Fallback** | Respuesta generica cuando el bot falla |
| **SSH** | Protocolo para conectarse al servidor desde tu computadora |
