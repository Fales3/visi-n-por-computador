import cv2
import cv2.aruco as aruco
import glob
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

import segment_funcs.img_funcs as img_f
import aruco.aruco_funcs as ar_f
import models.yolo_funcs as yolo_f

def get_aruco_board(aruco_dict, size: tuple= (5,7), square_length: int =30, marker_length: int =20 ):
    board = aruco.CharucoBoard(
        size,
        square_length,
        marker_length,
        aruco_dict
    )
    return board

def generate_charuco_pdf():
    # --- Parámetros del tablero ---
    squares_x = 5      # número de casillas horizontales
    squares_y = 7      # número de casillas verticales
    square_length_mm = 30   # tamaño de cada casilla (mm)
    marker_length_mm = 20   # tamaño del marcador ArUco dentro (mm)
    aruco_dict, parameters= ar_f.get_aruco_dict()

    # --- Crear tablero ---
    board = get_aruco_board(aruco_dict, (squares_x, squares_y), square_length_mm ,marker_length_mm)

    # --- Generar imagen del tablero ---
    # Aumentamos resolución para buena calidad
    img_size = (int(squares_x * square_length_mm * 15),  # ancho aproximado en píxeles
                int(squares_y * square_length_mm * 15))  # alto aproximado
    board_image = board.generateImage(img_size)

    # --- Guardar imagen temporal ---
    temp_img_path = "./images/charuco_temp.png"
    cv2.imwrite(temp_img_path, board_image)

    # --- Crear PDF A4 con escala real ---
    pdf_path = "charuco_A4_real_scale.pdf"
    page_width = 210 * mm
    page_height = 297 * mm
    c = canvas.Canvas(pdf_path, pagesize=(page_width, page_height))

    # Calcular tamaño real del tablero en mm
    board_width_mm = squares_x * square_length_mm
    board_height_mm = squares_y * square_length_mm

    # Centrar tablero en la hoja
    x_offset = (210 - board_width_mm) / 2 * mm
    y_offset = (297 - board_height_mm) / 2 * mm

    # Dibujar imagen a escala real
    c.drawImage(temp_img_path, x_offset, y_offset,
                width=board_width_mm * mm,
                height=board_height_mm * mm)

    c.showPage()
    c.save()

    print(f"✅ PDF generado correctamente: {pdf_path}")
    print(f"📏 Cada casilla mide {square_length_mm} mm en tamaño real")

def calibrate_charuco(dir_name: str):
    # --- Parámetros del tablero ---
    squares_x = 5      # número de casillas horizontales
    squares_y = 7      # número de casillas verticales
    square_length = 0.029   # tamaño de cada casilla (mm)
    marker_length = 0.019   # tamaño del marcador ArUco dentro (mm)
    aruco_dict, parameters= ar_f.get_aruco_dict()

    # --- Crear tablero ---
    board = get_aruco_board(aruco_dict, (squares_x, squares_y), square_length ,marker_length)

    # --- Variables para acumular detecciones ---
    all_corners = []
    all_ids = []
    image_size = None

    # --- Procesar todas las imágenes del tablero ---
    images = glob.glob(dir_name)  # Cambia según tu ruta
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        if len(corners) > 0:
            cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=None)
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if ret > 10:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                if image_size is None:
                    image_size = gray.shape[::-1]

    print(f"Usando {len(all_corners)} imágenes para calibrar")

    # --- Calibración ---
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners,
        all_ids,
        board,
        image_size,
        None,
        None
    )

    print("Error RMS:", ret)
    print("Matriz de cámara:\n", cameraMatrix)
    print("Coeficientes de distorsión:\n", distCoeffs)
    np.savez("charuco_calibration.npz", cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)

def load_calibration(filename: str):
    data = np.load(filename)
    camera_matrix = data["cameraMatrix"]
    dist_coeffs = data["distCoeffs"]

    print("Matriz de cámara:\n", camera_matrix)
    print("Coeficientes de distorsión:\n", dist_coeffs)

    return camera_matrix, dist_coeffs

def detect_charuco_pose(img, camera_matrix, dist_coeffs):
    aruco_dict, parameters= ar_f.get_aruco_dict()
    charuco_board=get_aruco_board(aruco_dict, (5, 7), 0.029 ,0.019)
    detector = cv2.aruco.CharucoDetector(charuco_board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img)

    if charuco_ids is not None and len(charuco_ids) > 3:
        success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            charuco_board,
            camera_matrix,
            dist_coeffs,
            None,
            None
        )

        if success:
            print("Pose estimada:")
            print("rvec:", rvec)
            print("tvec:", tvec)

            # Dibuja el eje del tablero
            cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
            img_f.show_img(img)

            return rvec, tvec
    else:
        print("No se detectaron suficientes esquinas ChArUco.")
        return None, None

def detect_aruco_pose(img, camera_matrix, dist_coeffs):
    aruco_dict, parameters = ar_f.get_aruco_dict()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(img)

    if ids is not None and len(ids) > 0:
        marker_length = 0.049  # tamaño real del marcador (en metros)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        # Elección del primer marcador como marcador base
        base_idx = 0
        rvec = rvecs[base_idx]
        tvec = tvecs[base_idx]

        # Dibuja los ejes de todos los marcadores detectados
        for i in range(len(ids)):
            cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)
            print(f"ID: {ids[i][0]} → tvec = {tvecs[i].ravel()} m")

        img_f.show_img(img)

        return rvec, tvec

    else:
        print("No se detectó marcador.")
        return None, None

    
def process_img_with_pose(img_name: str, filename: str,  use_charuco: False):
    camera_matrix, dist_coeffs= load_calibration(filename)
    img= img_f.select_img(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if use_charuco:
        rvec, tvec= detect_charuco_pose(img, camera_matrix, dist_coeffs)
    else:
        rvec, tvec= detect_aruco_pose(img, camera_matrix, dist_coeffs)

    model=yolo_f.load_yolo_model()
    results= yolo_f.load_img_to_model_calibrated(img_name, model, camera_matrix, dist_coeffs, rvec, tvec)
    img_f.show_img(results.plot(show=False))

    

