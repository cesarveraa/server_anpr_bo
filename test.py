import requests
import json

url = "http://localhost:8000/auto"
image_path = "placa.png"  # Reemplaza con la ruta de tu imagen

try:
    with open(image_path, "rb") as image_file:
        files = {"file": ("placa.png", image_file, "image/png")}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        response_json = response.json()
        print("Respuesta JSON estructurada del auto:")
        print(json.dumps(response_json, indent=4, ensure_ascii=False))
    else:
        print("Error:", response.status_code)
        try:
            print("Detalles del error:", response.json())
        except Exception as e:
            print("No se pudo decodificar la respuesta JSON:", str(e))

except FileNotFoundError:
    print(f"El archivo de imagen '{image_path}' no se encontró. Verifica la ruta y el nombre del archivo.")
except Exception as e:
    print("Error inesperado:", str(e))
