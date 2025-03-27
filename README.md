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
Here is an overview of the directory structure for this project:

``` bash
main/
│
├── api/                                   # FastAPI backend for image processing and prediction
│   └── fast.py                            # Main API logic and endpoints
│
├── interface/                             # Interface-related files for handling interactions
│   └── main.py                            # Main interface logic
│
├── ml_logic/                              # Machine Learning logic
│   └── modeling.py                        # Machine learning model handling and inference
│
├── notebooks/                             # Jupyter notebooks for model training and testing
│   ├── Spacegem_final_API_notebook.ipynb  # Final integration for API with models
│   ├── YOLO_model.ipynb                   # Notebook for YOLO model training and evaluation
│   ├── baseline-model.ipynb               # Baseline model notebook for initial experiments
│   └── models_integration.ipynb           # Notebook for integrating multiple models
│
├── .envrc                                 # Environment configuration file
├── .gitignore                             # Git ignore file to exclude unnecessary files
├── Dockerfile                             # Docker container setup for the project
├── Makefile                               # Build automation file (for setting up project tasks)
├── README.md                              # Project README (this file)
├── README_backup.md                       # Backup of the original README file
└── requirements.txt                       # Python dependencies for the project

```
***Explanation:***
1. api/:

Contains the backend logic for the project. It has:

- fast.py: Main logic for FastAPI routing and handling API requests for image classification and prediction.

2. interface/:

Contains the frontend code built with Streamlit, which provides the user interface for image uploads and displaying results.

- main.py: Handles interactions and communicates with the backend API.

3. ml_logic/:

Contains the machine learning logic for the project:

- modeling.py: Manages the loading and inference of machine learning models (YOLO for detection and CNN for classification).

4. notebooks/:

- Contains Jupyter notebooks for exploring, experimenting, and evaluating the models. It includes:

- Notebooks for training models and evaluating them, API building and testing

5. Configuration Files:

- .envrc: For environment-specific configurations

- .gitignore: Specifies which files and directories should not be tracked by Git

- Dockerfile: Setup instructions for containerizing the app.

- Makefile: Contains instructions to automate tasks such as setting up the environment, testing, or running scripts.

- requirements.txt: Specifies all Python dependencies for this project

6. Project Documentation:

7. README.md: A detailed file explaining the project.

## Data Security
***Image Storage:***
Images uploaded by users are processed in memory and not stored on the server to ensure privacy.

***Security Considerations:***
- The API is protected using FastAPI’s built-in validation to prevent misuse.
- The application does not store any personal data, ensuring compliance with data protection regulations.

## Credits
Guillaume Roland
Github username:Guillaume1987R

Jérôme Biot
Github username:Jayy-B

Sam
Github username:Mocopie

Shaked Keidar
Github username:CptSAmerica

Vinodha Ravichandran
Github username: Vinodhabiz

Special thanks to the Kaggle and Roboflow community for the dataset.
