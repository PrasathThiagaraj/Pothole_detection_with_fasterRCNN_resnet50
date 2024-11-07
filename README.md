<p align="center">Pothole_detection_with_fasterRCNN_resnet50</p>

**Pothole Detection Using Faster R-CNN**

This project is an automated system for real-time pothole detection using the Faster R-CNN deep learning model. The goal is to enhance road safety and improve infrastructure maintenance by accurately detecting potholes across varied environmental conditions. This repository contains code, models, and resources for training, testing, and deploying a pothole detection model on road images.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Introduction

Potholes are a common problem in road infrastructure, posing safety risks and increasing maintenance costs. Traditional methods rely on manual inspections, which are time-consuming and resource-intensive. This project leverages Faster R-CNN, a Region-based Convolutional Neural Network, for efficient and accurate pothole detection from road images, making real-time detection feasible under varied environmental conditions.

## Features
- **Automated Detection**: Real-time detection of potholes from images, reducing dependency on manual inspection.
- **High Accuracy**: Trained Faster R-CNN model with precision optimized for pothole identification.
- **Diverse Environmental Conditions**: Works across varied lighting and weather conditions.
- **Potential for Smart Transportation**: Useful for integration with autonomous vehicles, drones, or road-monitoring systems.

## Dataset
A dataset of road images, annotated for potholes, was used to train and validate the model. The images include varied conditions, such as different lighting, weather, and road surfaces.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/PrasathThiagaraj/Pothole_detection_with_fasterRCNN_resnet50
    cd pothole-detection
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have a compatible version of PyTorch installed:
    ```bash
    pip install torch torchvision
    ```

## Usage

1. **Training**: To train the Faster R-CNN model on your dataset, use:
    ```bash
    python train.py --data_dir /path/to/dataset --epochs 50
    ```

2. **Inference**: To run the model on new images for pothole detection:
    ```bash
    python detect.py --image_path /path/to/image.jpg
    ```

3. **Evaluation**: Evaluate the model’s accuracy, precision, and recall on test data:
    ```bash
    python evaluate.py --test_dir /path/to/testset
    ```

## Results
The trained model achieved an accuracy of approximately 90% under optimal conditions, with performance varying under low-light settings. Improvements are being made to enhance robustness across all environmental conditions.

## Future Work
Future enhancements include:
- Improving detection in low-light and challenging weather conditions.
- Incorporating additional data sources (e.g., LiDAR, accelerometers) to enhance detection robustness.
- Optimizing for deployment on edge devices, such as drones and autonomous vehicles.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Contributors**: Prasath T, Rohan John Ashok, Shreehari S


