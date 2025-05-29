import os
import re
import json
import time
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64

import openai
from openai import RateLimitError, OpenAIError
from dotenv import load_dotenv

# — Configuración de logging —
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# — Carga de variables de entorno desde .env —
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Falta la variable de entorno OPENAI_API_KEY")

# — Inicializa el cliente de OpenAI —
client = openai.OpenAI(api_key=api_key)

# — Inicializa FastAPI y CORS —
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://demo-anpr-bo.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Servidor FastAPI en Vercel funcionando correctamente"}

@app.post("/auto")
async def analyze_auto_image(file: UploadFile = File(...)):
    # Validar tipo de archivo
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")

    # Leer y codificar la imagen a data URL
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image_format = image.format.lower()
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    data_url = f"data:image/{image_format};base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            # Llamada a la API de OpenAI con visión
            completion = client.chat.completions.create(
                model="gpt-4o",  # Modelo vision-enabled de OpenAI
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze the image and provide a structured JSON response with the "
                                    "following information. If you're not sure about any field in "
                                    "`license_plate`, put \"not detected\". Return ONLY the JSON object:"
                                ),
                            },
                            {
                                "type": "text",
                                "text": """
{
  "car": {"make": "Car brand", "model": "Car model", "color": "Car color"},
  "license_plate": {"number": "License plate number", "country": "Country", "region": "Region or state"},
  "environment": {"location_type": "Street, parking lot, highway", "weather": "Sunny, rainy, cloudy", "time_of_day": "Day, night, evening"},
  "additional_info": {"occupants_visible": "Yes or No", "damage": "Any visible damage"}
}
""",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url}
                            }
                        ]
                    }
                ],
                temperature=0.1,
                top_p=0.1,
                max_tokens=800
            )

            # Obtener contenido crudo
            raw_content = completion.choices[0].message.content
            logger.info(f"Raw OpenAI response: {repr(raw_content)}")

            # Limpiar fences Markdown si existen
            cleaned = raw_content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                # eliminar fence de apertura
                lines = lines[1:]
                # eliminar fence de cierre si existe
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines)

            # Parsear JSON limpio
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as e:
                logger.error(f"JSONDecodeError: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error parsing cleaned JSON: {e}. Cleaned content: {repr(cleaned)}"
                )

            return JSONResponse(content=parsed)

        except RateLimitError:
            logger.warning(f"RateLimitError, intento {attempt}/{max_retries}")
            if attempt == max_retries:
                raise HTTPException(status_code=429, detail="Rate limit excedido, inténtalo más tarde.")
            time.sleep(2 ** attempt)

        except OpenAIError as e:
            logger.error(f"OpenAIError: {e}")
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

    # Tras todos los intentos fallidos
    raise HTTPException(status_code=500, detail="No se pudo procesar la imagen tras varios intentos.")
