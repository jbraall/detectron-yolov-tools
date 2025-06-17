"""
corregir_nombres_file_name.py

🛠 Este script corrige el campo `"file_name"` en archivos COCO JSON individuales,
para que coincida con los nombres reales de las imágenes en una carpeta.

🔧 ¿Qué problema soluciona?
Durante la conversión de anotaciones (por ejemplo, de PAGE XML a COCO),
es común que se asignen nombres de imagen inexactos, genéricos o derivados de forma incorrecta,
lo que puede hacer que el modelo no encuentre la imagen correspondiente.

🎯 ¿Qué hace exactamente?
- Busca todos los archivos `.jpg` en la carpeta de imágenes (`ruta_imagenes`).
- Crea un diccionario usando el **sufijo** del nombre de cada imagen (ignorando el prefijo numérico).
- Recorre todos los archivos `.json` en la subcarpeta `page/`.
- Para cada JSON, reemplaza el campo `"file_name"` con el nombre real de la imagen, si encuentra coincidencia de sufijo.

📌 Este enfoque es útil si las imágenes están nombradas como `00123_nombre.jpg`, `00087_nombre.jpg` y los JSON solo contienen `nombre.jpg`.

📥 Entrada esperada:
- Carpeta con imágenes (`.jpg`) correctamente nombradas.
- Carpeta con anotaciones COCO JSON, cada una con un solo `"images": [...]`.

📤 Salida:
- Se sobrescriben los archivos JSON originales con el nuevo `"file_name"` corregido.

✅ Este paso asegura que los nombres en el JSON y las imágenes coincidan exactamente, evitando errores de carga durante entrenamiento o validación.

Uso:
    python corregir_nombres_file_name.py
"""

import os
import json
from tqdm import tqdm

# === RUTAS ===
ruta_imagenes = "../../vorau_253_coco/train"
ruta_jsons = os.path.join(ruta_imagenes, "page")

# Indexar nombres reales de imágenes por sufijo
imagenes_dict = {}
for nombre in os.listdir(ruta_imagenes):
    if nombre.lower().endswith(".jpg"):
        sufijo = "_".join(nombre.split("_")[1:])  # quita el prefijo numérico
        imagenes_dict[sufijo] = nombre

# Procesar todos los JSON en labels/
for nombre_json in tqdm(os.listdir(ruta_jsons), desc="Corrigiendo file_name"):
    if not nombre_json.endswith(".json"):
        continue

    ruta_json = os.path.join(ruta_jsons, nombre_json)

    with open(ruta_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    imagen_info = data["images"][0]
    antiguo = imagen_info["file_name"]

    # Buscar una imagen cuyo sufijo coincida
    nuevo = imagenes_dict.get(antiguo)
    if nuevo:
        imagen_info["file_name"] = nuevo
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:
        print(f"⚠️ No se encontró imagen para: {antiguo}")

print("✅ Todos los JSON han sido corregidos (si había coincidencia).")
