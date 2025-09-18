import cv2
import os
import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T

import segment_funcs.img_funcs as img_f
def load_mask_r_cnn_model():
    
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    return model

def transform_img(img_name: str):
    transform = T.Compose([T.ToTensor()])
    img_f.select_img(img_name)
    img = Image.open(os.environ.get("IMAGE_PATH")).convert("RGB")
    img_tensor = transform(img)

    return img, img_tensor

def extract_mask(model, img_name):
    img, img_tensor = transform_img(img_name)
    with torch.no_grad():
        predictions = model([img_tensor])

    scores = predictions[0]['scores'].cpu().numpy()
    masks = predictions[0]['masks'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    img_np = np.array(img).copy()  # <--- inicializamos aquí

    for i in range(len(scores)):
        if scores[i] > 0.7:
            mask = masks[i, 0]
            bin_mask = mask > 0.5
            img_np[bin_mask] = [0, 255, 0]  # pintamos en verde

    plt.imshow(img_np)
    plt.axis("off")
    plt.show()
