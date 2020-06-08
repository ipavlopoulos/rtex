import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.models import Model
from keras.applications.densenet import preprocess_input
from keras.preprocessing import image
from tqdm import tqdm


def load_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    img.close()
    return x

def encode_images(cases_images, images_path):
    x1_data, x2_data = [], []

    for case in cases_images:
        images = cases_images[case].split(";")

        x1 = load_image(os.path.join(images_path, images[0]))
        x2 = load_image(os.path.join(images_path, images[1]))


        x1_data.append(x1)
        x2_data.append(x2)

    return [np.array(x1_data), np.array(x2_data)]
    
def extract_img_embeddings(model, images_path, data):
    case_vectors = {}
    for report in tqdm(data):
        images = data[report].split(";")
        encoded = []
        for i in images:
            # Encode image
            image_path = os.path.join(images_path, i)
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            encoded.append(x)
            img.close()

        # Predict
        x1 = np.expand_dims(encoded[0], axis=0)
        x2 = np.expand_dims(encoded[1], axis=0)
        vector = model.predict([x1,x2])

        case_vectors[report] = vector.transpose().flatten()
    return case_vectors
