"""
convert_pagexml_to_coco.py

Este script convierte archivos PAGE XML (utilizados en anotaciones de documentos históricos, OCR, etc.)
a formato COCO JSON compatible con frameworks de detección como Detectron2 o YOLOv8.

🚀 Funcionalidades principales:
- Recorre recursivamente un directorio fuente, manteniendo su estructura, y convierte los archivos .xml a .json.
- Copia también imágenes u otros archivos (no XML) al directorio destino.
- Extrae información de regiones de texto (`TextRegion`) y genera anotaciones COCO con bounding boxes y segmentaciones.
- Permite detectar la clase de cada región usando múltiples estrategias robustas:
  1. `custom="type:..."` → preferido si está presente.
  2. Texto en `TextEquiv/Unicode`, reconociendo etiquetas tipo `$par:`, `$tip:`, etc.
  3. Si no encuentra clase, asigna por defecto `"$pac"` para no descartar datos.

🧠 Manejo de situaciones complejas:
- Si una región no tiene `Coords`, puntos inválidos o demasiados pocos, se omite con advertencia.
- Si no se puede identificar una clase de forma confiable, se marca con `"$pac"` y se registra en un archivo `output.txt`.
- Se permite procesar XMLs que no contienen `imageFilename`, infiriéndolo del nombre del archivo.
- Permite convertir archivos aunque algunos contengan errores parciales (tolerancia a fallos).

📝 Archivos de salida:
- Un JSON COCO por cada archivo XML, con una imagen única (ID=1), sus anotaciones y categorías.
- Un archivo `output.txt` en la raíz que indica los archivos con regiones clasificadas de forma dudosa.

✅ Ideal para:
- Procesar datasets con etiquetas ruidosas o incompletas.
- Preparar datos para entrenamiento de modelos de detección/segmentación en documentos manuscritos.

Uso:
    python convert_pagexml_to_coco.py ruta/entrada ruta/salida
"""



#!/usr/bin/env python3
import os
import shutil
import json
import argparse
import re
import xml.etree.ElementTree as ET
from datetime import datetime

def parse_points(points_str):
    coords = []
    for point in points_str.split():
        try:
            x, y = point.split(',')
            coords.extend([float(x), float(y)])
        except Exception as e:
            print(f"Advertencia: error al parsear el punto '{point}': {e}")
    return coords

def convert_page_xml_to_coco(xml_file):
    ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    try:
        tree = ET.parse(xml_file)
    except Exception as e:
        raise ValueError(f"Error al parsear el XML '{xml_file}': {e}")
    root = tree.getroot()
    
    page = root.find('ns:Page', ns)
    if page is None:
        raise ValueError(f"El XML '{xml_file}' no contiene un elemento Page con el namespace esperado.")
    
    imageFilename = page.attrib.get("imageFilename")
    if imageFilename is None:
        imageFilename = os.path.splitext(os.path.basename(xml_file))[0] + ".jpg"
        print(f"Advertencia: No se encontró 'imageFilename' en Page, se derivó: {imageFilename}")
    imageWidth = page.attrib.get("imageWidth")
    imageHeight = page.attrib.get("imageHeight")
    if imageWidth is None or imageHeight is None:
        raise ValueError(f"El XML '{xml_file}' no tiene 'imageWidth' o 'imageHeight' en Page.")
    try:
        imageWidth = int(imageWidth)
        imageHeight = int(imageHeight)
    except Exception as e:
        raise ValueError(f"Error al convertir dimensiones en '{xml_file}': {e}")
    
    image = {
        "id": 1,
        "file_name": imageFilename,
        "width": imageWidth,
        "height": imageHeight,
        "license": 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": ""
    }
    
    fallo_econtrado = False
    annotations = []
    categories_dict = {}
    annotation_id = 1

    regions = page.findall('ns:TextRegion', ns)
    for region in regions:
        custom = region.attrib.get("custom", "")
        cat_name = None

        # Buscar clase en atributo custom y agregar "$"
        m = re.search(r'type:([^;}]+)', custom)
        if m:
            raw_type = m.group(1).strip()
            cat_name = raw_type if raw_type.startswith("$") else f"${raw_type}"

        # Si no hay clase en custom, buscar en TextEquiv/Unicode el $tipo:
        if not cat_name:
            text_equiv = region.find('ns:TextEquiv', ns)
            if text_equiv is not None:
                unicode_elem = text_equiv.find('ns:Unicode', ns)
                if unicode_elem is not None and unicode_elem.text:
                    m = re.search(r'(\$tip|\$nop|\$not|\$pag|\$par|\$pac):', unicode_elem.text)
                    if m:
                        cat_name = m.group(1)

        
        #hago el cambio y pongo qu ese ponga pac porque creo que es la que mas esta fallado. asi luego ya lo corregiré.
        # # Si aún no se ha encontrado clase, descartar la región
        # if not cat_name:
        #     print(f"Advertencia: Región {region.attrib.get('id', '')} sin clase identificable; se omitirá.")
        #     continue
        if not cat_name:
            fallo_econtrado = True
            cat_name = "$pac"  # Asignar etiqueta por defecto


        if cat_name not in categories_dict:
            categories_dict[cat_name] = len(categories_dict) + 1
        category_id = categories_dict[cat_name]

        coords_elem = region.find('ns:Coords', ns)
        if coords_elem is None:
            print(f"Advertencia: La región {region.attrib.get('id','')} no tiene Coords; se omitirá.")
            continue
        points_str = coords_elem.attrib.get("points", "")
        if not points_str:
            print(f"Advertencia: La región {region.attrib.get('id','')} tiene Coords sin puntos; se omitirá.")
            continue
        polygon = parse_points(points_str)
        if len(polygon) < 4:
            print(f"Advertencia: La región {region.attrib.get('id','')} tiene pocos puntos; se omitirá.")
            continue

        xs = polygon[0::2]
        ys = polygon[1::2]
        xmin = min(xs)
        ymin = min(ys)
        xmax = max(xs)
        ymax = max(ys)
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        area = bbox_width * bbox_height

        annotation = {
            "id": annotation_id,
            "image_id": 1,
            "category_id": category_id,
            "bbox": [xmin, ymin, bbox_width, bbox_height],
            "area": area,
            "iscrowd": 0,
            "segmentation": [polygon]
        }
        annotations.append(annotation)
        annotation_id += 1

    categories = []
    for cat_name, cat_id in categories_dict.items():
        categories.append({
            "id": cat_id,
            "name": cat_name,
            "supercategory": "none"
        })

    coco_dict = {
        "info": {
            "description": "Dataset converted from PAGE XML to COCO",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat()
        },
        "images": [image],
        "annotations": annotations,
        "categories": categories
    }
    
    return coco_dict, fallo_econtrado

def process_directory(source_dir, dest_dir):
    for root_dir, dirs, files in os.walk(source_dir):
        rel_path = os.path.relpath(root_dir, source_dir)
        dest_subdir = os.path.join(dest_dir, rel_path)
        os.makedirs(dest_subdir, exist_ok=True)

        for file in files:
            src_file_path = os.path.join(root_dir, file)
            ext = os.path.splitext(file)[1].lower()
            if ext == ".xml":
                try:
                    coco_json, fallo_encontrado_resumen = convert_page_xml_to_coco(src_file_path)
                    dest_file_name = os.path.splitext(file)[0] + ".json"
                    dest_file_path = os.path.join(dest_subdir, dest_file_name)
                    with open(dest_file_path, 'w', encoding='utf-8') as f:
                        json.dump(coco_json, f, ensure_ascii=False, indent=4)
                    print(f"Convertido: {src_file_path} -> {dest_file_path}")
                    if fallo_encontrado_resumen:
                        with open('output.txt', 'a', encoding='utf-8') as f:
                            f.write(f"Problem: {dest_file_path}\n")
                except Exception as e:
                    print(f"Error al procesar {src_file_path}: {e}")
            else:
                try:
                    shutil.copy2(src_file_path, os.path.join(dest_subdir, file))
                    print(f"Copiado: {src_file_path} -> {os.path.join(dest_subdir, file)}")
                except Exception as e:
                    print(f"Error al copiar {src_file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Convierte archivos PAGE XML a JSON (COCO) y copia la estructura de directorios."
    )
    parser.add_argument("source_dir", help="Directorio fuente que contiene los archivos (xml, jpg, etc.)")
    parser.add_argument("dest_dir", help="Directorio destino donde se creará la copia con la misma estructura, reemplazando los xml por coco json")
    args = parser.parse_args()

    source_dir = args.source_dir
    dest_dir = args.dest_dir

    if not os.path.exists(source_dir):
        print(f"El directorio fuente '{source_dir}' no existe.")
        return

    os.makedirs(dest_dir, exist_ok=True)
    process_directory(source_dir, dest_dir)

if __name__ == "__main__":
    main()
