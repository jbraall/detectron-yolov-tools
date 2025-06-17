"""
convertir_coco_a_yolo.py

📦 Este script convierte anotaciones de un archivo COCO JSON (detección sin máscaras)
al formato YOLO (una línea por objeto, con coordenadas normalizadas entre 0 y 1).

Formato de salida (por línea en cada .txt):
    class_id  x_center  y_center  width  height

🧩 ¿Qué hace exactamente?
- Lee un archivo COCO `.json` con campos `images`, `annotations`, `categories`.
- Reasigna las clases COCO a índices numéricos consecutivos (como espera YOLO).
- Para cada anotación:
    - Convierte el bbox de COCO `[x_min, y_min, w, h]` a YOLO `[x_c, y_c, w, h]` **normalizados**.
- Escribe un `.txt` por imagen con todas sus anotaciones.
- Guarda los `.txt` en la carpeta de salida correspondiente (`output_dir`).

🎯 Casos de uso:
- Preparar datasets anotados en COCO para entrenar modelos YOLOv5 o YOLOv8.
- Convertir datos generados automáticamente (por ejemplo desde PAGE XML → COCO) al formato YOLO.

🛠️ Consideraciones:
- Las imágenes deben coincidir con los nombres que figuran en `file_name` del COCO.
- No se generan carpetas de imágenes, solo los `.txt`.

📤 Salida:
- Un archivo `.txt` por imagen en la carpeta especificada, compatible con YOLOv5/YOLOv8.

📌 Este script está preparado para llamar directamente a la conversión de train / val / test,
pero puedes modificarlo fácilmente para un solo archivo o integrar en tu pipeline.

Uso:
    python convertir_coco_a_yolo.py
"""
#!/usr/bin/env python3
"""
COCO → YOLO (detección pura, sin máscaras)
• Una línea por objeto: class  x_c  y_c  w  h   (todos normalizados 0-1)
• Llama a la función tres veces (train / val / test).
"""

import json
import os
from pathlib import Path
from typing import Union, Dict, List


def convert_coco_json_to_yolo(coco_json_path: Union[str, Path],
                              output_dir: Union[str, Path]) -> None:
    """Convierte un único .json COCO a ficheros .txt estilo YOLO-v5/8 (5 columnas)."""
    coco_json_path = Path(coco_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Cargar el JSON -------------------------------------------------------
    with coco_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images: Dict[int, dict] = {img["id"]: img for img in data["images"]}
    categories: Dict[int, int] = {cat["id"]: idx  # mapea id COCO → nº continuo
                                  for idx, cat in enumerate(data["categories"])}

    # Dict para acumular líneas por imagen y escribir cada .txt de una vez
    yolo_lines: Dict[str, List[str]] = {}

    # -------------------------------------------------------------------------
    for ann in data["annotations"]:
        img = images[ann["image_id"]]
        w_img, h_img = img["width"], img["height"]

        # bbox COCO → YOLO normalizado
        x_min, y_min, w_box, h_box = ann["bbox"]
        x_c = (x_min + w_box / 2) / w_img
        y_c = (y_min + h_box / 2) / h_img
        w_n = w_box / w_img
        h_n = h_box / h_img

        class_id = categories[ann["category_id"]]
        line = f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"

        # Añadimos la línea al buffer de su imagen
        txt_name = Path(img["file_name"]).with_suffix(".txt").name
        yolo_lines.setdefault(txt_name, []).append(line)

    # --- Guardar todos los .txt ------------------------------------------------
    for txt_name, lines in yolo_lines.items():
        (output_dir / txt_name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"✓ {coco_json_path.name}: conversión completada → {output_dir}")


if __name__ == "__main__":
    # Modifica con tus rutas reales (.json incluido)
    convert_coco_json_to_yolo(
        coco_json_path="../../vorau_253_coco/train/json_unificado",
        output_dir="../../vorau_253_coco/yolov/labels/train"
    )
    convert_coco_json_to_yolo(
        coco_json_path="../../vorau_253_coco/val/json_unificado",
        output_dir="../../vorau_253_coco/yolov/labels/val"
    )
    convert_coco_json_to_yolo(
        coco_json_path="../../vorau_253_coco/test/json_unificado",
        output_dir="../../vorau_253_coco/yolov/labels/test"
    )
