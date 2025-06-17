"""
evaluador_unificado.py

Este script permite evaluar automáticamente los resultados de detección de objetos
en formato COCO para modelos Detectron2 o YOLOv8. Se le indica el tipo de modelo
con el argumento --modelo (detectron o yolo), y se encarga de:

- (Si es YOLO) Arreglar los image_id de las predicciones.
- Ejecutar la evaluación usando pycocotools.
- Guardar las métricas generales, el AP por clase y el Recall por clase en archivos JSON.

Ejemplo de uso:
  python evaluador_unificado.py --modelo yolo --gt ruta/a/gt.json --dt ruta/a/preds.json --save ruta/salida
"""


import argparse
import json
import os
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib
import io

def fix_image_ids(gt_path, dt_path, out_path):
    gt = json.load(open(gt_path))
    name2id = {Path(img["file_name"]).stem: img["id"] for img in gt["images"]}
    dt = json.load(open(dt_path))
    fixed, dropped = [], 0
    for ann in dt:
        stem = str(ann["image_id"])
        if stem in name2id:
            ann["image_id"] = name2id[stem]
            fixed.append(ann)
        else:
            dropped += 1
    print(f"[Fix YOLO] convertidos={len(fixed)}, descartados={dropped}")
    json.dump(fixed, open(out_path, "w"))

def evaluar(gt_json, dt_json, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(dt_json)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # === Métricas generales ===
    metricas_nombres = [
        "AP@[.5:.95]", "AP@0.5", "AP@0.75",
        "AP pequeña", "AP mediana", "AP grande",
        "AR@1", "AR@10", "AR@100",
        "AR pequeña", "AR mediana", "AR grande"
    ]
    metricas_valores = coco_eval.stats.tolist()

    with open(os.path.join(save_dir, "metricas_generales.json"), "w") as f:
        json.dump(dict(zip(metricas_nombres, metricas_valores)), f, indent=4)

    # === AP por clase ===
    print("\n📋 AP por clase (mAP@[.5:.95]):")
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_names = [c["name"] for c in cats]
    ap_por_clase = {}
    prec = coco_eval.eval["precision"]
    for idx, name in enumerate(cat_names):
        ap = prec[:, :, idx, 0, 2]
        ap = ap[ap > -1]
        ap_score = ap.mean() if ap.size else float("nan")
        ap_por_clase[name] = round(ap_score * 100, 2)
        print(f"{name:20}: {ap_por_clase[name]}")
    with open(os.path.join(save_dir, "ap_por_clase.json"), "w") as f:
        json.dump(ap_por_clase, f, indent=4)

    # === Recall por clase ===
    print("\n📋 Recall por clase (AR@100):")
    ar_por_clase = {}
    rec = coco_eval.eval["recall"]
    recalls = []
    for idx, name in enumerate(cat_names):
        ar = rec[:, idx, 0, 2]
        ar = ar[ar > -1]
        ar_score = ar.mean() if ar.size else float("nan")
        ar_por_clase[name] = round(ar_score * 100, 2)
        if not isinstance(ar_score, float) or not (ar_score != ar_score):
            recalls.append(ar_score)
        print(f"{name:20}: {ar_por_clase[name]}")
    recall_medio = round((sum(recalls) / len(recalls)) * 100, 2) if recalls else 0.0
    ar_por_clase["Recall Medio"] = recall_medio
    print(f"\n📌 Recall medio general: {recall_medio}")
    with open(os.path.join(save_dir, "recall_por_clase.json"), "w") as f:
        json.dump(ar_por_clase, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluador unificado para COCO (YOLOv8 / Detectron2)")
    parser.add_argument("--modelo", type=str, choices=["detectron", "yolo"], required=True, help="Modelo usado: detectron o yolo")
    parser.add_argument("--gt", type=str, required=True, help="Ruta al archivo JSON de anotaciones (GT)")
    parser.add_argument("--dt", type=str, required=True, help="Ruta al archivo JSON de predicciones")
    parser.add_argument("--save", type=str, default="./resultados_eval", help="Ruta donde guardar las métricas")
    args = parser.parse_args()

    if args.modelo == "yolo":
        fixed_path = os.path.join(args.save, "predictions_fixed.json")
        fix_image_ids(args.gt, args.dt, fixed_path)
        evaluar(args.gt, fixed_path, args.save)
    else:
        evaluar(args.gt, args.dt, args.save)
