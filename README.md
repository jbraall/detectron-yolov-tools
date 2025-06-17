# 🧰 Herramientas de Procesamiento y Validación de Datasets (COCO, YOLO)

Este repositorio contiene scripts desarrollados para facilitar el procesamiento, validación y conversión de datasets anotados en formato COCO. Están organizados según el modelo al que están orientados: **Detectron2**, **YOLOv8** y herramientas de evaluación general.

---

## 📊 `benchmark_tools/`

- **`evaluador_unificado.py`**  
  Evalúa predicciones de modelos en formato COCO (Detectron2 o YOLOv8), generando métricas globales y por clase, incluyendo:
  - AP@[.5:.95], AP por clase
  - AR@100, Recall por clase
  - Archivos JSON exportados con resultados

---

## 🔧 `detectron2_tools/`

- **`conocer_numero_parametros.py`**  
  Muestra el número total de parámetros de un modelo Detectron2 (ej. Mask R-CNN con ResNet-50-FPN).

- **`corregir_dimensiones_coco.py`**  
  Compara las dimensiones registradas en el JSON COCO con las reales de las imágenes, y las corrige si hay errores.

- **`corregir_nombres_file_name.py`**  
  Corrige los nombres de archivo (`file_name`) en archivos COCO para que coincidan con los nombres reales de las imágenes en disco.

- **`unificar_anotaciones_coco.py`**  
  Une múltiples archivos COCO (uno por imagen) en un único archivo con IDs únicos y categorías unificadas. Necesario para entrenamiento en Detectron2.

- **`visualizar_anotaciones_coco.py`**  
  Dibuja las anotaciones COCO (bounding boxes, clases) sobre las imágenes originales y guarda una copia etiquetada para verificación visual.

- **`voc_to_coco_last_update.py`**  
  Convierte anotaciones en formato Pascal VOC XML al formato COCO. Útil para migrar datasets antiguos.

---

## 🚀 `yolov8_tools/`

- **`convertir_coco_a_yolo.py`**  
  Convierte anotaciones desde COCO a YOLOv5/YOLOv8 (`.txt` por imagen, con coordenadas normalizadas).

- **`visualizar_segmentaciones_yolo.py`**  
  Visualiza anotaciones YOLO con segmentaciones poligonales, dibujando bounding boxes y máscaras sobre las imágenes.

---

## 📎 Notas

- Todos los scripts están diseñados para ser usados de forma independiente.
- Algunas herramientas requieren que las rutas de entrada/salida se modifiquen manualmente en el script.
- Recomendado usar entorno virtual y tener dependencias como `opencv-python`, `tqdm`, `numpy`, `Pillow`, `pycocotools`, `detectron2`.

---

## ✍️ Autor

Este repositorio forma parte de un conjunto de aportaciones técnicas desarrolladas con el objetivo de facilitar el entrenamiento y evaluación de modelos de detección de instancias en documentos manuscritos.

---
