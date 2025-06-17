"""
visualizar_anotaciones_coco.py

🧩 Este script carga un dataset en formato COCO (anotaciones + imágenes), y guarda una copia
de cada imagen con las anotaciones dibujadas (bounding boxes y clases). Así puedes revisar fácilmente
si las etiquetas están correctas o si hay errores en la anotación.

✅ ¿Qué hace?
- Registra el dataset con `register_coco_instances` desde Detectron2.
- Carga las imágenes y sus anotaciones (desde el JSON unificado).
- Dibuja las cajas (y segmentaciones si las hay) sobre cada imagen original.
- Guarda las imágenes etiquetadas en una carpeta de salida.

🎯 Casos de uso:
- Verificar visualmente la calidad de un dataset antes de entrenar.
- Validar que el script de conversión XML → COCO ha funcionado bien.
- Mostrar ejemplos de entrenamiento para informes, papers o documentación.

🛠️ Requisitos:
- Tener Detectron2 instalado.
- El archivo JSON debe estar en formato COCO y unificado.
- Las rutas deben estar correctamente definidas en las variables:
    - `json_annotations` → anotaciones COCO
    - `ruta_imagenes` → imágenes originales
    - `output_dir` → carpeta de salida con las imágenes anotadas

📌 El script también maneja errores de carga de imágenes y crea la carpeta de salida si no existe.

Uso:
    python visualizar_anotaciones_coco.py
"""
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import os
import cv2
from tqdm import tqdm

# === CONFIGURACIÓN ===
nombre_dataset = "mi_dataset_train"
json_annotations = "../../vorau_253_coco/train/json_unificado"
ruta_imagenes = "../../vorau_253_coco/train/"
output_dir = "../../vorau_253_coco/etiquetadas_train"  # Carpeta donde guardar las imágenes etiquetadas

# === REGISTRO DEL DATASET ===
register_coco_instances(nombre_dataset, {}, json_annotations, ruta_imagenes)

# Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# === CARGA DE DATOS ===
dataset_dicts = DatasetCatalog.get(nombre_dataset)
metadata = MetadataCatalog.get(nombre_dataset)

# === PROCESAR Y GUARDAR ===
for d in tqdm(dataset_dicts, desc="Etiquetando imágenes"):
    img = cv2.imread(d["file_name"])
    if img is None:
        print(f"⚠️ No se pudo cargar: {d['file_name']}")
        continue

    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out = v.draw_dataset_dict(d)
    imagen_etiquetada = out.get_image()[:, :, ::-1]

    # Extraer solo el nombre del archivo
    nombre_archivo = os.path.basename(d["file_name"])
    ruta_salida = os.path.join(output_dir, nombre_archivo)

    cv2.imwrite(ruta_salida, imagen_etiquetada)

print("✅ Imágenes etiquetadas guardadas en:", output_dir)