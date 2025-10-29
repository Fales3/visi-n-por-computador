import cv2
import numpy as np
import segment_funcs.img_funcs as img_f
import segment_funcs.segmentacion as seg
"""
YOLOv8-seg:                 rápido y ligero, precisión media.
Mask R-CNN:                 mayor precisión, más pesado y lento.
Segment Anything (SAM):     pesado, segmentación universal, necesita identificación mediante prompt o click
U-Net:                      Fotos individuales
"""

def load_yolo_model():
    from ultralytics import YOLO

    model = YOLO("yolo_models/yolo11m-seg.pt")

    return model

def load_img_to_model(img_name, model, factor_px_cm= 1):
    
    img=img_f.select_img(img_name)
    results = model(img)

    for i, mask in enumerate(results[0].masks.data):
        # convertir tensor a imagen binaria
        mask_np = mask.cpu().numpy().astype("uint8") * 255

        # 2. Reescalar la máscara al tamaño original
        mask_np = cv2.resize(
            mask_np, 
            (img.shape[1], img.shape[0]),  # ancho, alto de la original
            interpolation=cv2.INTER_NEAREST
        )
        
        # encontrar contornos
        contours=img_f.find_contours(mask_np)
        img_contours=cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 5)
        
        if len(contours) > 0:
            cnt = contours[0]
            seg.measurement_with_contours(img_contours, cnt, factor_px_cm)

    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # coordenadas de la caja
            conf = box.conf[0]
            cls = int(box.cls[0])
            print(f"Objeto detectado: {model.names[cls]}, confianza: {conf:.2f}")
            print(f"Caja: {x1}, {y1}, {x2}, {y2}")

    return results[0]

def load_img_to_model_calibrated(img_name, model, camera_matrix, dist_coeffs, rvec, tvec):
    img = img_f.select_img(img_name)
    results = model(img)

    for i, mask in enumerate(results[0].masks.data):
        mask_np = mask.cpu().numpy().astype("uint8") * 255
        mask_np = cv2.resize(mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        contours = img_f.find_contours(mask_np)
        img_contours = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
        img_f.show_img(img_contours)

        if len(contours) > 0:
            cnt = contours[0]

            #Conversión de puntos a coordenadas reales
            cnt_undist = cv2.undistortPoints(
            cnt.astype(np.float32),
            camera_matrix,
            dist_coeffs,
            P=camera_matrix
    )

            seg.measurements_with_pose(img_contours, cnt_undist, camera_matrix, rvec, tvec)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            print(f"Objeto detectado: {model.names[cls]}, conf: {conf:.2f}")
            print(f"Caja: {x1}, {y1}, {x2}, {y2}")

    return results[0]
