import cv2
import os

import aruco.aruco_funcs as ar_f
import segment_funcs.img_funcs as img_f

def classic_segment_img(img_name:str, is_aruco: bool= True, blurring_method: str="median", threshold_method: str="OTSU", show: bool= True):
    
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

    img_f.select_img(img_name)
    img=cv2.imread(os.environ.get("IMAGE_PATH"))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=img_f.img_to_grayscale(img)
    if is_aruco:
        img_rgb=ar_f.detect_aruco(img, img_rgb)
    img=img_f.blur_img(img, blurring_method)
    img=img_f.threshold_img(img,threshold_method)
    img=img_f.morphology(img)
    contours=img_f.find_contours(img)

    img_contours=img_rgb.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

    if show is True:
        img_f.show_img(img_contours)


def segment_with_aruco(img_name: str):

    img_f.select_img(img_name)
    img=cv2.imread(os.environ.get("IMAGE_PATH"))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=img_f.img_to_grayscale(img)
    img_rgb=ar_f.detect_aruco(img, img_rgb)