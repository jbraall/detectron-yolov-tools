"""
visualizar_segmentaciones_yolo.py

🧩 Este script permite visualizar las anotaciones YOLOv8 (formato con segmentación) sobre las imágenes originales,
dibujando tanto las cajas como las máscaras. Genera una copia de cada imagen con las etiquetas superpuestas
para facilitar la validación visual del dataset.

🎯 ¿Qué hace?
- Lee las imágenes desde un directorio.
- Lee los archivos `.txt` con anotaciones YOLO:
    - Primera parte: class_id x_center y_center width height (normalizado)
    - Parte opcional: coordenadas poligonales (segmentación) normalizadas.
- Dibuja:
    - Bounding boxes con la clase.
    - Máscaras si están presentes (polígonos).
- Guarda las imágenes resultantes en una carpeta de salida.

📥 Requiere:
- `image_dir`: carpeta con las imágenes originales.
- `label_dir`: carpeta con archivos `.txt` en formato YOLOv8 con segmentaciones opcionales.
- `output_dir`: carpeta donde se guardarán las imágenes etiquetadas.
- `class_names` (opcional): lista de nombres de clase.

📌 Funciones útiles:
- Usa colores distintos por clase generados aleatoriamente.
- Verifica que las etiquetas y las imágenes correspondan.
- Ignora anotaciones mal formadas o incompletas.

✅ Este script es muy útil para:
- Verificar visualmente datasets segmentados para YOLOv8.
- Detectar errores de anotación antes de entrenar.
- Generar ejemplos para documentación o informes.

Uso:
    python visualizar_segmentaciones_yolo.py
"""
import os
import cv2
import random
import numpy as np

def get_class_colors(n_classes):
    random.seed(42)
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]

def draw_yolo_segmentations(image_dir, label_dir, output_dir, class_names=None):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    n_classes = len(class_names) if class_names else 10
    class_colors = get_class_colors(n_classes)

    for img_file in image_files:
        image_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error al leer la imagen: {image_path}")
            continue

        h, w, _ = image.shape

        if not os.path.exists(label_path):
            print(f"No hay etiqueta para {img_file}, se omite.")
            continue

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls = int(float(parts[0]))
                x_c, y_c, bw, bh = map(float, parts[1:5])
                poly_coords = list(map(float, parts[5:]))

                color = class_colors[cls % len(class_colors)]
                label = class_names[cls] if class_names and cls < len(class_names) else str(cls)

                # Dibujar bounding box
                x1 = int((x_c - bw / 2) * w)
                y1 = int((y_c - bh / 2) * h)
                x2 = int((x_c + bw / 2) * w)
                y2 = int((y_c + bh / 2) * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Dibujar máscara de segmentación
                if poly_coords:
                    points = np.array(poly_coords, dtype=np.float32).reshape(-1, 2)
                    points[:, 0] *= w
                    points[:, 1] *= h
                    points = points.astype(np.int32)
                    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
                    cv2.fillPoly(image, [points], color=color + (80,))  # transparencia (opcional)

        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, image)

    print(f"Imágenes con segmentación guardadas en: {output_dir}")


# 🧪 USO DEL SCRIPT:
draw_yolo_segmentations(
    image_dir="../../vorau_253_coco/yolov/images/val",
    label_dir="../../vorau_253_coco/yolov/labels/val",
    output_dir="../../vorau_253_coco/yolov/imagenes_etiquetadas/val",
    class_names=["lyrics", "staff", "drop-capital"]
)
