import cv2
import numpy as np
from keras.models import load_model


# Classify fresh/rotten
def print_fresh(res):
    threshold_fresh = 0.10  # set according to standards
    threshold_medium = 0.35  # set according to standards
    print_fresh(res[0][0])  # Print the freshness status
    if res < threshold_fresh:
        print("The item is FRESH!")
    elif threshold_fresh < res < threshold_medium:
        print("The item is MEDIUM FRESH")
    else:
        print("The item is NOT FRESH")


def pre_proc_img(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess the image
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def evaluate_rotten_vs_fresh(image_path):
    # Load the trained model
    model = load_model(r'C:\Users\shiva\Desktop\Github\Green Grub\extra\trained-freshness-model.h5')

    # Read and process and predict
    prediction = model.predict(pre_proc_img(image_path))
    
    return prediction[0][0]


