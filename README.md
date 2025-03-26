Project Overview
Project Name: Spacegem (Space Gemstone Classifier)

# Space Gem 🚀
**Space Gem** is a machine learning-powered gemstone classification app that leverages Convolutional Neural Networks (CNN) and the YOLO model to predict and identify various types of gemstones from images. It offers a user-friendly interface to upload gemstone images, analyze them, and get gemstone predictions in real-time.


## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation Instructions](#installation-instructions)
3. [API Usage](#api-usage)
4. [Model Details](#model-details)
5. [Frontend Usage](#frontend-usage)
6. [Project Structure](#project-structure)
7. [Data Security](#data-security)
8. [Credits](#credits)

## Project Overview

Space Gem uses a **YOLO model** to identify and count gemstones in an uploaded image. When only one gemstone is recognized, a **CNN model** is used for classification. The YOLO model recognizes five types of gemstones, while the CNN model supports 87 gemstone classifications. The app provides a detailed prediction along with a description of the gemstone through OpenAPI integration.

- **Frontend**: Developed with **Streamlit** for easy interaction.
- **Backend**: Uses **FastAPI** and **YOLO** for prediction.
- **Training Data**: Sourced from Kaggle (link will be updated).

## Installation Instructions

To set up the Space Gem app on your local machine:

### Prerequisites
Ensure you have the following tools installed:

- **Python 3.7+**
- **Docker** (for containerization)
- **Streamlit** for frontend deployment
- **FastAPI** for backend API

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/spacegem.git
   cd spacegem

2. Install dependencies:
```bash
pip install -r requirements.txt
 ```

3. Run the backend FastAPI app:
 ``` bash
 uvicorn api.fast:app --reload
  ```

4. Start the frontend using Streamlit:
``` bash
streamlit run space_gem/app.py
```

### API Usage
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
-Both models were trained using data sourced from Kaggle. (Kaggle dataset URL).

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
│   ├── model_v2_1.keras         # Trained CNN model version 1
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

Special thanks to the Kaggle community for the dataset.
