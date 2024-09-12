# Brain Tumor Detection and Segmentation using ResUNet

This project leverages a ResUNet model to classify whether a patient has a brain tumor and segments the tumor from the brain image. The model is trained on medical imaging data, and the segmentation helps in identifying the precise location of the tumor.

## Features
- **Classification**: The model predicts whether the patient has cancer based on brain scans.
- **Segmentation**: Accurately segments and visualizes the tumor region in the brain images.

## How to Run the Project

### 1. Train the Segmentation Model
To train the model for brain tumor segmentation, run the following command:

```bash
python models/train_segmentation.py
```
### 2. View the Tumor Segmentation
Once the model is trained, you can view the segmentation results by running:

```bash
python main.py
```
