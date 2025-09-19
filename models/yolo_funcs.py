import cv2
import os

import segment_funcs.img_funcs as img_f

"""
YOLOv8-seg:                 rápido y ligero, precisión media.
Mask R-CNN:                 mayor precisión, más pesado y lento.
Segment Anything (SAM):     pesado, segmentación universal, necesita identificación mediante prompt o click
U-Net:                      Fotos individuales
"""

def load_yolo_model():
    from ultralytics import YOLO

    # Modelo preentrenado (puedes empezar con coco)
    model = YOLO("yolov8s-seg.pt")

    return model

def load_img_to_model(img_name, model):

    img_f.select_img(img_name)
    img=cv2.imread(os.environ.get("IMAGE_PATH"))
    results = model(img)

    masks = results[0].masks #Info de las máscaras
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # coordenadas de la caja
            conf = box.conf[0]
            cls = int(box.cls[0])
            print(f"Objeto detectado: {model.names[cls]}, confianza: {conf:.2f}")
            print(f"Caja: {x1}, {y1}, {x2}, {y2}")

    results[0].plot(show=True)

def get_yolo_format(path_name: str):
    #Rutas de entrada
    img_dir_train = path_name+"images/train"
    mask_dir_train = path_name+"masks/train"
    img_dir_val = path_name+"images/val"
    mask_dir_val = path_name+"masks/val"

    #Rutas de salida
    out_lbl_train = path_name+"labels/train"
    out_lbl_val = path_name+"labels/val"

    for d in [out_lbl_train, out_lbl_val]:
        os.makedirs(d, exist_ok=True)

    process_split(img_dir_train, mask_dir_train, out_lbl_train)
    process_split(img_dir_val, mask_dir_val, out_lbl_val)

def process_split(img_dir: str, mask_dir: str, out_lbl_dir: str):
    for fname in os.listdir(img_dir):
        if not fname.endswith(".jpg"):
            continue

        # Generar label desde máscara
        mask_path = os.path.join(mask_dir, fname)
        label_path = os.path.join(out_lbl_dir, fname.replace(".jpg", ".txt"))
        if os.path.exists(mask_path):
            mask_to_yolo(mask_path, label_path)

def mask_to_yolo(mask_path: str, label_path: str, class_id=0):
    mask=cv2.imread(mask_path, 0)
    contours=img_f.find_contours(mask)
    cnt = max(contours, key=cv2.contourArea)
    h, w = mask.shape
    poly = []
    for point in cnt:
        x, y = point[0]
        poly.append(f"{x/w:.6f}")
        poly.append(f"{y/h:.6f}")

    # Guardar en archivo .txt
    with open(label_path, "w") as f:
        f.write(f"{class_id} " + " ".join(poly) + "\n")