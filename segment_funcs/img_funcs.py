import os
import cv2

def select_img(file_name:str)->None:
    img_path= f"./images/{file_name}"

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"La imagen {file_name} no existe")
    
    os.environ["IMAGE_PATH"]=img_path

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

def threshold_img(img, method):
    if method=="adaptative":
        print("Método adaptativo")
        th = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 7
        )
    elif method=="OTSU":
        print("Método OTSU")
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError("Método inválido")

    return th

def morphology(img):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    th = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return th

def refine_contour(c):
    epsilon = 0.01 * cv2.arcLength(c, True)  # 1% de la longitud del perímetro
    approx = cv2.approxPolyDP(c, epsilon, True)
    return approx

def find_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [refine_contour(c) for c in contours if cv2.contourArea(c) > 2000]

    return contours

def show_img(img):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.title("Imagen")
    plt.axis('off')
    plt.show()