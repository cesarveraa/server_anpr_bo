import os
import openai
import json
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64

# Configurar la clave de API de SambaNova
os.environ["SAMBANOVA_API_KEY"] = "56c33ee7-d76e-47a3-be76-8ddd525f997f"

# Inicializar el cliente de SambaNova
client = openai.OpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

app = FastAPI()

# Endpoint to receive an image and send it to Llama Vision
@app.post("/auto")
async def analyze_auto_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")

    # Read the image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image_format = image.format.lower()
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image_format)
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')

    # Agregar el prefijo adecuado al base64 para Llama Vision
    image_base64 = f"data:image/{image_format};base64," + image_base64

    # Send the image to Llama Vision with a structured prompt
    max_retries = 5
    retry_delay = 5  # segundos
    attempt = 0

    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model="Llama-3.2-90B-Vision-Instruct",
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "Analyze the image and provide a structured JSON response with the following information, if your are not sure about any of the fields in license_plate, do not fill it and put: not detected. Ensure the response is a valid JSON object only, without additional explanation:"
                    }, {
                        "type": "text",
                        "text": """
{
    "car": {
        "make": "Car brand",
        "model": "Car model",
        "color": "Car color",
        "type": "Sedan, SUV, Truck, etc.",
        "year": "Estimated year of the car"
    },
    "license_plate": {
        "number": "License plate number",
        "country": "Country of the license plate",
        "region": "Region or state of the license plate, if identifiable"
    },
    "environment": {
        "location_type": "Street, parking lot, highway, etc.",
        "weather": "Sunny, rainy, cloudy, etc.",
        "time_of_day": "Day, night, evening, etc."
    },
    "additional_info": {
        "occupants_visible": "Are there people visible inside the car?",
        "damage": "Any visible damage to the vehicle",
        "accessories": "Visible accessories (e.g., roof rack, decals, modifications)"
    }
}
"""
                    }, {
                        "type": "image_url",
                        "image_url": {"url": image_base64}
                    }]
                }],
                temperature=0.1,
                top_p=0.1
            )

            # Imprimir la respuesta completa para depuración
            print("Respuesta de Llama Vision:", response)

            # Validar la estructura de la respuesta
            if response and response.choices and len(response.choices) > 0:
                json_response = response.choices[0].message.content
                try:
                    # Intentar convertir la respuesta a JSON de forma segura
                    parsed_response = json.loads(json_response)
                    return JSONResponse(content=parsed_response)
                except json.JSONDecodeError as e:
                    raise HTTPException(status_code=500, detail=f"Error parsing JSON response: {str(e)}")
            else:
                raise HTTPException(status_code=500, detail="Respuesta inválida de Llama Vision.")

        except openai.RateLimitError as e:
            attempt += 1
            if attempt < max_retries:
                print(f"Rate limit exceeded. Reintentando en {retry_delay} segundos... (Intento {attempt}/{max_retries})")
                time.sleep(retry_delay)
                continue
            else:
                raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error communicating with Llama Vision: {str(e)}")
