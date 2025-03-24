from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import os
from io import BytesIO
import numpy as np
import json
from tensorflow.keras.preprocessing import image
import uvicorn

app = FastAPI()

model_path = os.path.dirname(os.path.dirname(__file__))
absolute_path = os.path.join(model_path, "raw_data", "model_v2_3.keras")
#absolute_path = os.path.join(model_path, "raw_data", "model_v2_1.keras")

#absolute_path = "/home/vinodha/code/CptSAmerica/Space_Gem/raw_data/model_v2_225.keras"
# Load the model from the .keras file shared by Jerome
#absolute_path = "model_v2_225.keras"
model = load_model(absolute_path)
#model.summary()

# Preprocessing function for the uploaded image
def preprocess_image(image_bytes):
    #img = image.load_img(BytesIO(image_bytes), target_size=(225, 225))
    img = image.load_img(BytesIO(image_bytes), target_size=(225, 225))
    img_array = image.img_to_array(img)
    #img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 225, 225, 3)
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

# the prediction API endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img = await file.read()  # Get the image bytes from the uploaded file
    predicted_label = get_gemstone_label(img)  # Get predicted gemstone label
    return {"predicted_gemstone": predicted_label}
