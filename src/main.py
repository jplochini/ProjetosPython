import os
import numpy as np
import pandas as pd
from skimage import io
from sklearn.model_selection import train_test_split
import cv2

def pre_processing_images(dataset):
    imagens_pre_processadas = []

    for image_path in os.listdir(dataset):
        image = cv2.imread(os.path.join(dataset, image_path))

        #redimensionar a imagem para tamanho fixo
        image = cv2.resize(image, (150, 150))

        #converter a imagem para escala de cinza
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #normalizar a imagem
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        imagens_pre_processadas.append(image)

    return imagens_pre_processadas
