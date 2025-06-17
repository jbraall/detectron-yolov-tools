"""
unificar_anotaciones_coco.py

Este script toma múltiples archivos COCO JSON (uno por imagen) dentro de una carpeta
y los combina en un único archivo COCO válido, con IDs únicos y categorías unificadas.
Este formato unificado es necesario para frameworks como Detectron2, que requieren
un solo archivo de anotaciones para todo el dataset.

🔧 Qué hace específicamente:
- Recorre todos los JSONs dentro de una carpeta (`input_folder`).
- Reasigna los `image_id` y `annotation_id` para que no haya conflictos.
- Unifica las `categories` por nombre (aunque los IDs originales difieran).
- Reescribe los `category_id` en cada anotación con un nuevo ID global.
- Crea un único archivo COCO (`output_file`) listo para entrenamiento con Detectron2.

🎯 Casos de uso:
- Datasets anotados individualmente por imagen (por ejemplo, salidos de conversores PAGE XML → COCO).
- Situaciones en las que los archivos `.json` locales tienen conflictos de IDs o categorías.

📦 Entrada esperada:
- Carpeta con varios archivos `.json` estilo COCO, cada uno con una sola imagen y sus anotaciones.

📤 Salida:
- Un único archivo JSON COCO (`json_unificado`) que contiene:
    - Todas las imágenes.
    - Todas las anotaciones.
    - Categorías globales sin duplicados.

✅ Este script es esencial para preparar datos de anotación dispersos para su uso en modelos modernos de detección.

Uso:
    python unificar_anotaciones_coco.py
"""

import json
from pathlib import Path

# Ruta de entrada y salida
input_folder = "../../vorau_253_coco/train/page"
output_file = "../../vorau_253_coco/train/json_unificado"

# Acumuladores
all_images = []
all_annotations = []
all_categories = {}
image_id_counter = 1
annotation_id_counter = 1

# Contador para asignar nuevos category_id únicos
category_name_to_id = {}
category_id_counter = 1

# Leer todos los archivos
json_files = list(Path(input_folder).glob("*.json"))

for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Obtener categorías locales de este archivo (pueden tener ID duplicados)
    local_id_to_name = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    # Unificar categorías globalmente según su nombre
    for local_id, name in local_id_to_name.items():
        if name not in category_name_to_id:
            category_name_to_id[name] = category_id_counter
            all_categories[category_id_counter] = {"id": category_id_counter, "name": name, "supercategory": "none"}
            category_id_counter += 1

    # Reasignar imagen
    old_image_id = data["images"][0]["id"]
    new_image_id = image_id_counter
    data["images"][0]["id"] = new_image_id
    all_images.append(data["images"][0])
    image_id_counter += 1

    # Reasignar anotaciones con nuevo ID de categoría
    for ann in data["annotations"]:
        ann["id"] = annotation_id_counter
        ann["image_id"] = new_image_id

        # Corregir el category_id
        original_local_cat_id = ann["category_id"]
        category_name = local_id_to_name[original_local_cat_id]
        ann["category_id"] = category_name_to_id[category_name]

        all_annotations.append(ann)
        annotation_id_counter += 1

# Construcción del JSON final
final_data = {
    "info": {
        "description": "Dataset combinado para Detectron2 con categorías unificadas",
        "version": "1.0",
        "year": 2025
    },
    "images": all_images,
    "annotations": all_annotations,
    "categories": list(all_categories.values())
}

# Guardar archivo final
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=4)

print(f"✅ Dataset combinado y corregido guardado en: {output_file}")
