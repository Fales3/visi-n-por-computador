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
        _, th = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)
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
    contours = [refine_contour(c) for c in contours if cv2.contourArea(c) > 2000]

    return contours

def show_img(img):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.title("Imagen")
    plt.axis('off')
    plt.show()

