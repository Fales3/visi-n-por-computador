import cv2
import cv2.aruco as aruco

def get_aruco_dict():
    # Diccionario de marcadores
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    return aruco_dict, parameters

def generate_aruco(aruco_id: int=0, pixels: int= 200):
    # Diccionario de marcadores
    aruco_dict, parameters = get_aruco_dict()

    # Crear un marcador de 200x200 px
    marker = aruco.generateImageMarker(aruco_dict, aruco_id, pixels)

    cv2.imwrite("./images/aruco_marker.png", marker)

    return aruco_dict

def detect_aruco(img, img_rgb, real_side_cm: float= 4.9):
    import numpy as np

    aruco_dict, parameters= get_aruco_dict()

    detector = aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejected = detector.detectMarkers(img)

    print(f"Marcadores rechazados: {len(rejected)}")

    #corners: lista de esquinas de cada marcador detectado.
    #ids: lista con el ID numérico detectado.
    #rejected: candidatos descartados (cosas que parecían ArUco pero no lo eran).
    
    if ids is not None:
        for i, corner in enumerate(corners):
            # corners[i] tiene 4 puntos (x, y) del marcador
            pts = corner[0]

            lado1 = np.linalg.norm(pts[0] - pts[1])
            lado2 = np.linalg.norm(pts[1] - pts[2])
            lado3 = np.linalg.norm(pts[2] - pts[3])
            lado4 = np.linalg.norm(pts[3] - pts[0])

            lado_px = (lado1 + lado2 + lado3 + lado4) / 4


            # Tamaño real del marcador (cm)

            # Factor px/cm
            factor_px_cm = lado_px / real_side_cm

            print(f"ID: {ids[i][0]}")
            print(f"Esquinas del marcador:\n{pts}")
            print(f"Lados px: {lado1:.2f}, {lado2:.2f}, {lado3:.2f}, {lado4:.2f}")
            print(f"Lado promedio en px: {lado_px:.2f}")
            print(f"Factor px/cm: {factor_px_cm:.4f}")


            # Dibujar el marcador
            aruco.drawDetectedMarkers(img_rgb, corners, ids)

    else:
        print("No se detecó marcador ArUco")
            
    return img_rgb, factor_px_cm



