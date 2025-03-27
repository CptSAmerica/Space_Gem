from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import os
from io import BytesIO
import numpy as np
import json
from tensorflow.keras.preprocessing import image
import uvicorn
import requests
from ultralytics import YOLO


app = FastAPI()

model_path = os.path.dirname(os.path.dirname(__file__))
absolute_path = os.path.join(model_path, "models", "model_v2_3.keras")

model = load_model(absolute_path)

# Preprocessing function for the uploaded image
def preprocess_image(image_bytes):
    img = image.load_img(BytesIO(image_bytes), target_size=(225, 225))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load gemstone classes from the JSON file
json_path = os.path.join(model_path, "raw_data", "gemstone_classes.json")

with open(json_path, "r") as f:
    gemstone_classes = json.load(f)

# Function to map the prediction index to a gemstone label
def get_gemstone_label(image_bytes):
    processed_image = preprocess_image(image_bytes)
    print(f"Processed image shape: {processed_image.shape}")
    prediction = model.predict(processed_image)
    print (prediction)
    predicted_class_index = np.argmax(prediction[0])
    print(f"Predicted class index: {predicted_class_index}")
    # Using gemstone_classes list
    predicted_label = gemstone_classes[predicted_class_index]
    return predicted_label

@app.get("/")
def index():
    return {"status": "ok"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    We call this function first.
    It predicts objects in an image from a file.
    If multiple gems are recognized,
    the second model is being called.
    """

    space_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(space_path, "models", "best.onnx")

    img_path= os.path.join(space_path, "raw_data", 'input_image.jpg')
    with open(img_path, "wb") as buffer:
        buffer.write(await file.read())

    with open(img_path, "rb") as buffer:
        model_image_file = buffer.read()

    try:
        # Load the exported ONNX model
        onnx_model = YOLO(model_path)

        # Run inference
        results = onnx_model(img_path)

        # Count how many gems of each category appear
        detections = results[0].boxes.data.tolist()

        count_dict = {'Ruby': 0, 'Amethyst': 0, 'Diamond': 0, 'Emerald': 0, 'Sapphire': 0}
        for detection in detections:
            # The last element in each detection list is the class ID
            class_id = int(detection[-1])

            if class_id == 0:
                count_dict['Ruby'] += 1
            elif class_id == 1:
                count_dict['Amethyst'] += 1
            elif class_id == 2:
                count_dict['Diamond'] += 1
            elif class_id == 3:
                count_dict['Emerald'] += 1
            elif class_id == 4:
                count_dict['Sapphire'] += 1

        # Counting amout of recognized gems
        gem_count = (count_dict['Ruby'] + count_dict['Amethyst']
        + count_dict['Diamond'] + count_dict['Emerald'] + count_dict['Sapphire'])

        if gem_count > 1:
            results[0].show()
            return count_dict
        else:
            predicted_label = get_gemstone_label(model_image_file)
            return predicted_label

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
    except ValueError as e:
        print(f"Error determining image format: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
