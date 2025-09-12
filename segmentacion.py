import os

def select_img(file_name:str)->None:
    """
    Función que seleciciona y guarda en el environ la imagen a trabajar

    PARAMETROS:
    -----------
    file_name: str
        Nombre de la imagen a trabajar, con su extensión
    
    RETURN:
    -------
    None
    """
    img_path= f"./images/{file_name}"

    if not os.path.exist(img_path):
        raise FileNotFoundError(f"La imagen {file_name} no existe")
    
    os.environ["IMAGE_PATH"]=img_path

def img_to_grayscale()->None:
    """
    Función que convierte la imagen a escala de grises

    PARAMETROS:
    -----------
    None
    
    RETURN:
    -------
    None
    """

    import cv2
    import numpy

    imagen=cv2.imread(os.environ["IMAGE_PATH"])
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque
    blur = cv2.GaussianBlur(gris, (5, 5), 0)

    


    

