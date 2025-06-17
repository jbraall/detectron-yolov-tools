"""
Este script carga una configuración de Detectron2 (por ejemplo, Mask R-CNN con ResNet-50-FPN),
construye el modelo y calcula el número total de parámetros entrenables. Es útil para comparar
la complejidad del modelo con otros como YOLOv8-L.

Asegúrate de ajustar la ruta al archivo YAML de configuración y el número de clases según tu dataset.

Requiere que Detectron2 esté correctamente instalado y accesible.
"""

from detectron2.config import get_cfg
from detectron2.modeling import build_model

# Cargar configuración desde un YAML (puedes usar cualquier config oficial o personalizado)
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x2.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # o ajusta al número de clases de tu dataset

# Construir el modelo
model = build_model(cfg)

# Contar los parámetros
total_params = sum(p.numel() for p in model.parameters())
print(f"Total de parámetros: {total_params:,}")