# Pneumonia Detection using CNN and ResNet

This project focuses on detecting pneumonia from chest X-ray images using deep learning. Two models were implemented:

A custom Convolutional Neural Network (CNN) achieving 81% accuracy

A ResNet-based model achieving 68% accuracy

Overview
To enhance interpretability, LIME (Local Interpretable Model-agnostic Explanations) and Grad-CAM (Gradient-weighted Class Activation Mapping) were applied, providing visual insights into how the models made their predictions.

Dataset
Chest X-ray images were sourced from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia.
Images are classified as either Pneumonia or Normal.

Preprocessing
OpenCV was used to preprocess the images. This included removing text and enhancing image quality to improve model performance.

Data augmentation techniques were applied to address class imbalance, helping the model generalize better.

Models Used
Custom CNN
4 convolutional layers

Batch normalization & dropout for regularization

Adam optimizer & categorical cross-entropy loss

ResNet-based Model
Pretrained on ImageNet

Fine-tuned on pneumonia dataset

Used for comparison with custom CNN

Results
Model	Accuracy
Custom CNN	81%
ResNet	68%
Model Interpretability
LIME: Highlights the most important pixels influencing the model’s decision.

Grad-CAM: Visualizes activated regions in X-ray images.

Example Interpretability Images:
<img width="640" alt="image" src="https://github.com/user-attachments/assets/1461a3c3-d8c3-43e2-8dee-4e6402551c91" />
<img width="625" alt="image" src="https://github.com/user-attachments/assets/3446c990-78c8-4cb5-9b74-f4263dc8591c" />


Acknowledgments
Special thanks to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia and the deep learning research community for guidance and resources.
