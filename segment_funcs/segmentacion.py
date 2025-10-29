import cv2
import numpy as np
import aruco.aruco_funcs as ar_f
import models.yolo_funcs as yolo_f
import segment_funcs.img_funcs as img_f

def classic_segment_img(img_name:str, is_aruco: bool= False, blurring_method: str="median", threshold_method: str="OTSU", show: bool= True):
    
    """
    Función que segmenta una imagen, con o sin presencia de Aruco y puede mostrar la imagen final

    PARAMETERS
    ----------
    img_name: str
        Nombre de la imagen, con su extensión incluida
    aruco: bool
        Ingreso manual si hay presencia o no de ArUco
    blurring_method: str
        método para el suavizado de la imagen, puede ser 'gauss' o 'median'
    threshold_method: str
        método para la umbralización, puede ser 'adaptative' u 'OTSU'
    show: bool
        Ingreso manual si se desea motrar la imagen al finalizar o no.
    """

    img=img_f.select_img(img_name)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if is_aruco:
        img_rgb, lado_real_cm, factor_px_cm=ar_f.detect_aruco(img, img_rgb)
    gray=img_f.img_to_grayscale(img)
    # img=img_f.blur_img(img, blurring_method)
    img=img_f.threshold_img(gray,threshold_method)
    img=img_f.morphology(img)
    contours=img_f.find_contours(img)
    img_contours=img_rgb.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

    if show is True:
        img_f.show_img(img_contours)

def segment_with_aruco_and_yolo(img_name: str, white_background: bool= False, aruco_side_cm: float=5.0, show_img: bool= True):
    img= img_f.select_img(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb=img.copy()

    if not white_background:
        gray=img_f.img_to_grayscale(img)
        _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    img_rgb, factor_px_cm=ar_f.detect_aruco(img, img_rgb, real_side_cm=aruco_side_cm)

    if show_img:
        img_f.show_img(img_rgb)
    
    model=yolo_f.load_yolo_model()

    results=yolo_f.load_img_to_model(img_name, model, factor_px_cm)
    
    if show_img:
        img_f.show_img(results.plot(show=False))

def measurement_with_contours(img, cnt, factor_px_cm: float):
    area_px = cv2.contourArea(cnt)
    perimeter_px = cv2.arcLength(cnt, True)

    # bounding box
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect

    box = cv2.boxPoints(rect)   # obtiene las esquinas del rectángulo
    box = np.intp(box)          # convierte a enteros
    img_contours=img.copy()
    cv2.drawContours(img_contours, [box], 0, (0,0,255), 5)  # azul
    img_f.show_img(img_contours)
    
    print(f" Centroide (px): ({cx:.2f}, {cy:.2f}), ángulo: {angle:.2f}°")
    print(f" - Área en px: {area_px}")
    print(f" - Perímetro en px: {perimeter_px}")
    print(f" - Dimensiones (px): ancho={w}, alto={h}")

    img_sides=img.copy()
    # dibujar el lado w
    cv2.line(img_sides, tuple(box[0]), tuple(box[1]), (0, 0, 255), 5)  # azul
    # dibujar el lado h
    cv2.line(img_sides, tuple(box[1]), tuple(box[2]), (255, 0, 0), 5)  # rojo

    img_f.show_img(img_sides)

    if factor_px_cm is not None:
        print(f" - Área en cm²: {area_px / (factor_px_cm**2):.2f}")
        print(f" - Ancho en cm: {w/factor_px_cm:.2f}, Alto en cm: {h/factor_px_cm:.2f}")

def project_to_plane(u, v, camera_matrix, R, tvec):
    """Proyecta un punto (u,v) en píxeles al plano Z=0 del marcador en coordenadas reales."""
    uv1 = np.array([[u, v, 1.0]]).T
    ray = np.linalg.inv(camera_matrix) @ uv1

    n = R @ np.array([[0, 0, 1]]).T
    d = n.T @ tvec
    λ = -d / (n.T @ ray)

    X_c = λ * ray
    X_plane = R.T @ (X_c - tvec)
    return X_plane[:2]  # X, Y (en metros)

def measurements_with_pose(img, contour, camera_matrix, rvec, tvec):
    """
    Calcula ancho y alto reales (en cm) de un contorno proyectado sobre el plano calibrado.
    Usa minAreaRect sobre los puntos reales.
    """
    # --- Preparar rotación y traslación ---
    R, _ = cv2.Rodrigues(rvec) #vector de rotación a matriz de rotación
    tvec = tvec.reshape(3, 1) #aseguramos dimensión correcta

    # --- Proyectar todos los puntos del contorno ---
    real_contour = np.array([
        project_to_plane(float(u), float(v), camera_matrix, R, tvec).flatten()
        for [[u, v]] in contour
    ])

    # Convertir a centímetros
    real_contour_cm = real_contour * 100

    # --- Calcular bounding box real mínimo ---
    rect = cv2.minAreaRect(real_contour_cm.astype(np.float32))

    (width, height) = rect[1]
    width, height = max(width, height), min(width, height)
    print(f"Ancho: {width:.2f} cm | Alto: {height:.2f} cm")

    return width, height, rect, real_contour_cm
