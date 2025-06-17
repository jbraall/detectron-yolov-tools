"""
corregir_dimensiones_coco.py

🧩 Este script revisa y corrige las dimensiones (`width`, `height`) de cada imagen listada
en un archivo COCO JSON, asegurando que coincidan con las dimensiones reales del archivo de imagen.

🔧 ¿Qué hace exactamente?
- Abre cada imagen mencionada en el JSON.
- Obtiene sus dimensiones reales con PIL (Python Imaging Library).
- Compara con las dimensiones anotadas en el JSON.
- Si no coinciden, las corrige y guarda un nuevo JSON actualizado.

🎯 ¿Por qué es importante?
- Algunos conversores XML → COCO pueden generar anotaciones con dimensiones incorrectas.
- Detectron2 y otros frameworks **fallan si las dimensiones no coinciden exactamente**.
- Este script es útil como paso de validación antes de entrenar o evaluar.

📥 Entrada esperada:
- `json_path`: ruta al JSON COCO original con posibles errores de tamaño.
- `images_folder`: carpeta donde están almacenadas las imágenes reales.

📤 Salida:
- `output_json_path`: nuevo archivo JSON con los tamaños corregidos.

📌 El script también avisa si alguna imagen del JSON no se encuentra físicamente.

Uso:
    python corregir_dimensiones_coco.py
"""
import json
import os
from PIL import Image
from tqdm import tqdm

# RUTA AL JSON ORIGINAL Y A LAS IMÁGENES
json_path = "../../vorau_253_coco/detectron/test_json_unificado"
images_folder = "../../vorau_253_coco/detectron/test"
output_json_path = "../../vorau_253_coco/detectron/test_json_unificado2"

# Cargar el JSON
with open(json_path, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

# Crear un diccionario rápido para buscar por nombre
filename_to_image = {img["file_name"]: img for img in coco_data["images"]}

# Recorrer todas las imágenes del JSON
for img_info in tqdm(coco_data["images"], desc="Corrigiendo tamaños"):
    file_name = img_info["file_name"]
    img_path = os.path.join(images_folder, file_name)

    # Asegurarse de que la imagen existe
    if not os.path.exists(img_path):
        print(f"⚠️ Imagen no encontrada: {img_path}")
        continue

    # Obtener tamaño real de la imagen
    with Image.open(img_path) as img:
        width, height = img.size

    # Corregir en el JSON
    if img_info["width"] != width or img_info["height"] != height:
        print(f"🛠 Corrigiendo {file_name}: ({img_info['height']}, {img_info['width']}) → ({height}, {width})")
        img_info["width"] = width
        img_info["height"] = height

# Guardar el JSON corregido
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(coco_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ JSON corregido guardado en: {output_json_path}")
