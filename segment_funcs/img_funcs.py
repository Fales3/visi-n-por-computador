import os
import cv2

def select_img(file_name:str)->None:
    img_path= f"./{file_name}"

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"La imagen {file_name} no existe")
    
    img=cv2.imread(img_path)

    return img

def img_to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return gray

def blur_img(img, method:str):
    if method=="gauss":
        print("Método Gauss")
        img = cv2.GaussianBlur(img, (5,5), 0)
    elif method=="median":
        print("Método Median")
        img = cv2.medianBlur(img, 5)
    else:
        raise ValueError("Método inválido")

    return img

def threshold_img(img, method, maxval: float=255, thresh:float=0):
    if method=="adaptative":
        print("Método adaptativo")
        th = cv2.adaptiveThreshold(
            img, maxval,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 7
        )
    elif method=="OTSU":
        print("Método OTSU")
        _, th = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method=="simple":
        print("simple")
        _, th = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY_INV)
    else:
        raise ValueError("Método inválido")

    return th

def morphology(img):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    #Apertura: quita ruido pequeño
    th = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #Cierre: rellena huecos
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    return th

def refine_contour(c):
    epsilon = 0.01 * cv2.arcLength(c, True)  # 1% de la longitud del perímetro
    approx = cv2.approxPolyDP(c, epsilon, True)
    return approx

def find_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def show_img(img):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.title("Imagen")
    plt.axis('off')
    plt.show()

def verify_mask(filename: str):
    import numpy as np
    img_path="dataset/images/train/"+filename+".jpg"
    label_path="dataset/labels/train/"+filename+".txt"
    
    select_img(img_path)
    img=cv2.imread(os.environ.get("IMAGE_PATH"))

    h, w = img.shape[:2]
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            points = [(int(coords[i] * w), int(coords[i+1] * h)) for i in range(0, len(coords), 2)]
            cv2.polylines(img, [np.array(points)], isClosed=True, color=(0,255,0), thickness=2)

    # Mostrar
    show_img(img)