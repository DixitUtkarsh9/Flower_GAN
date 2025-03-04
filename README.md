# Flower Classification

## Overview
This project focuses on classifying three types of flowers using various deep learning techniques. The project includes multiple classification tasks, as well as image denoising and generative modeling using GANs.

## Tasks Performed
1. **Feedforward Neural Network (FNN)** - Implemented a simple neural network for classification.
2. **Custom Convolutional Neural Network (CNN)** - Built a CNN from scratch for improved accuracy.
3. **ResNet with Pretrained Weights** - Used a pretrained ResNet model to leverage transfer learning.
4. **Image Denoising** - Added noise to images and developed a denoising model.
5. **Generative Adversarial Network (GAN)** - Implemented a GAN model for generating synthetic flower images.

## Dataset
- The dataset consists of images of three different flower types.
- Preprocessing includes image resizing, normalization, and augmentation for better generalization.

## Technologies Used
- Python
- TensorFlow / PyTorch
- OpenCV
- NumPy & Pandas
- Matplotlib & Seaborn
- Jupyter Notebook / Google Colab

## Installation
To set up the project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/Flower-Classification.git
cd Flower-Classification

# Install dependencies
pip install -r requirements.txt
```

## Usage
To train the different models, run the corresponding scripts:

```bash
# Train Feedforward Neural Network
python train_fnn.py

# Train Custom CNN
python train_cnn.py

# Train ResNet Model
python train_resnet.py

# Train Image Denoising Model
python train_denoise.py

# Train GAN Model
python train_gan.py
```

## Results
| Model | Accuracy / Performance |
|--------|------------------------|
| Feedforward Neural Network | Basic classification accuracy |
| Custom CNN | Improved accuracy over FNN |
| ResNet (Pretrained) | High accuracy leveraging transfer learning |
| Denoising Model | Successfully reduced noise in images |
| GAN | Generated synthetic flower images |

## Future Enhancements
- Fine-tune models for better accuracy.
- Extend classification to more flower types.
- Develop a web application for real-time classification.
- Improve GAN model for higher-quality image generation.

