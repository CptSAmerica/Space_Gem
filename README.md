Project Overview
Project Name: Spacegem (Gemstone Classifier)

# Space Gem 🚀
**Space Gem** is a computer vision gemstone classification app that leverages Convolutional Neural Networks (CNN) and the YOLO model to predict and identify various types of gemstones from images. It offers a user-friendly interface to upload gemstone images and get them recognized in real-time.


## Table of Contents

1. [Project Overview](#project-overview)
2. [API Usage](#api-usage)
3. [Model Details](#model-details)
4. [Frontend Usage](#frontend-usage)
5. [Project Structure](#project-structure)
6. [Data Security](#data-security)
7. [Credits](#credits)

## Project Overview

Space Gem uses a **YOLO model** to identify and count gemstones in an uploaded image. When only one gemstone is recognized, a **CNN model** is used for classification. The YOLO model recognizes five types of gemstones, while the CNN model supports 87 gemstone classifications. The app provides a detailed prediction along with a description of the gemstone through OpenAPI integration.

- **Frontend**: Developed with **Streamlit** for easy interaction.
- **Backend**: Uses **CNN** and **YOLO** for prediction.
- **Training Data**:
    Sourced from Kaggle data set: `[Link](https://www.kaggle.com/datasets/lsind18/gemstones-images)`, Roboflow data set: `[Link](https://universe.roboflow.com/val-hphzo/gemstones-2e1jx/dataset/6)`


## API Usage
The Space Gem API is built using FastAPI and can be accessed by sending an image via a POST request.

### Endpoint
- **URL:** http://localhost:8000/upload/**
- **Method:** POST

### Form Data
image: Image file to be uploaded.

***Response Example***
```json
{
  "status": "ok",
  "predictions": [
    {
      "Ruby"
      "A ruby is a variety of corundum and is considered one of the four precious stones."
    },
  ]
}
```

### How to Call the API
You can test the API by uploading an image through the Streamlit frontend, which sends requests to the backend for prediction.

## Model Details
### YOLO Model:
- Recognizes 5 types of gemstones.
- Uses the YOLO architecture for object detection.

### CNN Model:
- Classifies 87 types of gemstones.
- Based on a convolutional neural network, fine-tuned for gemstone classification.

## Frontend Usage
1. Upload an Image: Visit the frontend interface built with Streamlit. Upload an image containing gemstones.

2. Prediction: The API processes the image and returns predictions, showing details about the gemstone type and description.

The user does not need to log in to use the service.

## Project Structure
Here is the structure of the Space Gem app:

``` bash
SPACE_GEM/
│
├── api/                         # FastAPI backend for image processing and prediction
│   ├── fast.py                  # Main API logic
│
├── notebooks/                   # Jupyter notebooks for model training and testing
│   ├── baseline-model.ipynb     # Baseline model notebook
│   ├── model_v2_3.keras         # Trained CNN model version 3
│   └── ...
│
├── raw_data/                    # Raw image data for training
│   ├── alexandrite_18.jpg       # Example gemstone image
│   └── ...
│
├── space_gem/                   # Streamlit frontend app
│   └── app.py                   # Streamlit UI logic
│
├── .gitignore                   # Git ignore file
├── Dockerfile                   # Docker container setup
├── requirements.txt             # Python dependencies
└── README.md                    # Project README
```
## Data Security
***Image Storage:***
Images uploaded by users are processed in memory and not stored on the server to ensure privacy.

***Security Considerations:***
- The API is protected using FastAPI’s built-in validation to prevent misuse.
- The application does not store any personal data, ensuring compliance with data protection regulations.

## Credits
"add you names and other details here"

Vinodha Ravichandran
Github username: Vinodhabiz

Special thanks to the Kaggle and Roboflow community for the dataset.
