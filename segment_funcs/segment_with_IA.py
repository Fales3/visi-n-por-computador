import os
import cv2
import segment_funcs.segmentacion as seg
import aruco.aruco_funcs as ar_f

def segment_img_IA(img_name: str):

    seg.select_img(img_name)
    img=cv2.imread(os.environ.get("IMAGE_PATH"))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=seg.img_to_grayscale(img)
    img_rgb=ar_f.detect_aruco(img, img_rgb)
    seg.show_img(img_rgb)

