# Video Classifier using Transfer Learning

This repository contains a deep learning project for video classification using the **ID3 dataset**. The model leverages **transfer learning** with pre-trained weights to classify video content into specific categories. Built using TensorFlow/Keras, this project demonstrates the power of transfer learning for efficient and accurate video classification.

## Features

- **Transfer Learning**
- **Video Classification**: Classifies video content into categories using the ID3 dataset.
- **Deep Learning**: Built with TensorFlow/Keras for high accuracy.
- **Easy to Use**: Includes scripts for training, evaluation, and prediction.

## Usage

1. Clone the repository:
  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
   

4. Evaluate the model:
   

5. Make predictions:
   ```bash
   python predict.py --input path/to/video
   ```

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- OpenCV
- Matplotlib

Example `requirements.txt`:
```plaintext
tensorflow>=2.0.0
numpy>=1.19.0
opencv-python>=4.5.0
matplotlib>=3.3.0
```

## Model Details

- **Transfer Learning**: Uses pre-trained weights from a deep learning model (e.g., ResNet, Inception, or EfficientNet) for feature extraction.
- **ID3 Dataset**: Trained on the ID3 dataset for video classification.
- **Custom Layers**: Added custom layers on top of the pre-trained model for fine-tuning.

## Example

Input:
```bash
python predict.py --input data/test_video.mp4
```

Output:
```
Predicted Class: Sports
```


