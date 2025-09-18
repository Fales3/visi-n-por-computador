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
    

    results[0].plot(show=True)