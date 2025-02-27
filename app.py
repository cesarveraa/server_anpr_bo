import os
import openai
import json
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Importar el middleware de CORS
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64

# Configurar la clave de API de SambaNova
os.environ["SAMBANOVA_API_KEY"] = "04ab4fe9-1716-4c13-8851-6d7589ae597e"

# Inicializar el cliente de SambaNova
client = openai.OpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

app = FastAPI()
# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://demo-anpr-bo.vercel.app"],  # Permitir el origen del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def read_root():
    return {"message": "Servidor FastAPI en Vercel funcionando correctamente"}
@app.post("/auto")
async def analyze_auto_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")

    try:
        # Read and process the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_format = image.format.lower()
        image_bytes = io.BytesIO()
        image.save(image_bytes, format=image_format)
        image_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
        image_base64 = f"data:image/{image_format};base64," + image_base64

        max_retries = 5
        attempt = 0
        while attempt < max_retries:
            try:
                response = client.chat.completions.create(
                    model="Llama-3.2-90B-Vision-Instruct",
                    messages=[{
                        "role": "user",
                        "content": [ 
                            {
                                "type": "text",
                                "text": "Analyze the image and provide a structured JSON response with the following information, if you are not sure about any of the fields in license_plate, do not fill it and put: not detected. Ensure the response is a valid JSON object only, without additional explanation:"
                            }, 
                            {
                                "type": "text",
                                "text": """
                                {
                                    "car": {"make": "Car brand","model": "Car model","color": "Car color"},
                                    "license_plate": {"number": "License plate number","country": "Country of the license plate","region": "Region or state of the license plate, if identifiable"},
                                    "environment": {"location_type": "Street, parking lot, highway","weather": "Sunny, rainy, cloudy","time_of_day": "Day, night, evening"},
                                    "additional_info": {"occupants_visible": "Yes or No","damage": "Any visible damage"}
                                }
                                """
                            }, 
                            {
                                "type": "image_url",
                                "image_url": {"url": image_base64}
                            }
                        ]
                    }],
                    temperature=0.1,
                    top_p=0.1
                )

                print("Respuesta completa de Llama Vision:", response)

                # Acceder directamente al contenido sin validaciones adicionales
                if response and response.choices and len(response.choices) > 0:
                    json_response = response.choices[0].message.content
                    print("Contenido de la respuesta:", json_response)
                    try:
                        # Convertir la respuesta a JSON de forma segura
                        parsed_response = json.loads(json_response)
                        return JSONResponse(content=parsed_response)
                    except json.JSONDecodeError as e:
                        print("Error al decodificar JSON:", str(e))
                        raise HTTPException(status_code=500, detail=f"Error parsing JSON response: {str(e)}")
                else:
                    print("Respuesta inválida de Llama Vision:", response)
                    raise HTTPException(status_code=500, detail="Respuesta inválida de Llama Vision.")

            except openai.RateLimitError as e:
                attempt += 1
                print(f"Rate limit exceeded. Reintentando... (Intento {attempt}/{max_retries})")
                time.sleep(5)
            except Exception as e:
                print("Error en la solicitud a Llama Vision:", str(e))
                raise HTTPException(status_code=500, detail=f"Error communicating with Llama Vision: {str(e)}")

    except Exception as e:
        print("Error general en el backend:", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")
