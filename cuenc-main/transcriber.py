"""Transcripcion de audio (Whisper) e imagenes (GPT-4o Vision) usando OpenAI API."""

import io
import base64
import logging
import httpx
from openai import AsyncOpenAI
from config import OPENAI_API_KEY, CHATWOOT_API_TOKEN

log = logging.getLogger(__name__)

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def _download_file(url: str) -> bytes | None:
    """Descarga un archivo desde una URL (con auth Chatwoot si es necesario)."""
    headers = {}
    if "chats.alef.company" in url or "chatwoot" in url.lower():
        headers["api_access_token"] = CHATWOOT_API_TOKEN
        # Chatwoot Active Storage: usar proxy en vez de redirect para evitar 404 en disk URLs
        url = url.replace("/blobs/redirect/", "/blobs/proxy/")

    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            log.info(f"Download response: {resp.status_code} ({len(resp.content)} bytes) url={url[:100]}")
            if resp.status_code == 200:
                return resp.content
            log.error(f"Error descargando archivo ({resp.status_code}): {url[:120]}")
    except Exception as e:
        log.error(f"Error descargando archivo: {e}")
    return None


async def transcribe_audio(audio_url: str) -> str | None:
    """Descarga y transcribe un audio usando Whisper."""
    audio_data = await _download_file(audio_url)
    if not audio_data:
        return None

    try:
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.ogg"

        transcript = await openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="es",
        )
        text = transcript.text.strip()
        log.info(f"Audio transcrito ({len(audio_data)} bytes): {text[:100]}")
        return text
    except Exception as e:
        log.error(f"Error transcribiendo audio: {e}")
        return None


async def describe_image(image_url: str) -> str | None:
    """Descarga y describe una imagen usando GPT-4o Vision."""
    image_data = await _download_file(image_url)
    if not image_data:
        return None

    try:
        b64 = base64.b64encode(image_data).decode("utf-8")
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe brevemente esta imagen en espanol. Si contiene texto, transcribelo. Si es un documento medico, receta o resultado de laboratorio, indica que tipo de documento es y resume su contenido. Maximo 3 lineas.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        text = response.choices[0].message.content.strip()
        log.info(f"Imagen descrita ({len(image_data)} bytes): {text[:100]}")
        return text
    except Exception as e:
        log.error(f"Error describiendo imagen: {e}")
        return None
