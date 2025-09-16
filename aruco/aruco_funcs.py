import cv2
import cv2.aruco as aruco

def get_aruco_dict():
    # Diccionario de marcadores
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    return aruco_dict, parameters

def generate_aruco():
    # Diccionario de marcadores
    aruco_dict = get_aruco_dict

    # Crear un marcador de 200x200 px
    marker = aruco.generateImageMarker(aruco_dict, 0, 200)

    cv2.imwrite("aruco_marker.png", marker)

    return aruco_dict

def detect_aruco(img, img_rgb):
    import numpy as np

    aruco_dict, parameters= get_aruco_dict()

    detector = aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejected = detector.detectMarkers(img)
    #corners: lista de esquinas de cada marcador detectado.
    #ids: lista con el ID numérico detectado.
    #rejected: candidatos descartados (cosas que parecían ArUco pero no lo eran).
    
    if ids is not None:
        for i, corner in enumerate(corners):
            # corners[i] tiene 4 puntos (x, y) del marcador
            pts = corner[0]

            lado_px = np.linalg.norm(pts[0] - pts[1])

            # Tamaño real del marcador (mm)
            lado_real_mm = 50.0  

            # Factor mm/px
            factor_mm_px = lado_real_mm / lado_px

            print(f"ID: {ids[i][0]}")
            print(f"Lado en píxeles: {lado_px:.2f}")
            print(f"Factor mm/px: {factor_mm_px:.4f}")

            # Dibujar el marcador
            print(f"ID: {ids[i]}, esquinas: {corner}")
            aruco.drawDetectedMarkers(img_rgb, corners, ids)
            
    return img_rgb

