import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


from models.loss import focal_tversky, tversky_loss, tversky
from models.resunet import ResUNet
from utils import prediction  
from utils.data_loader import BrainDataset
from models.resnet import ResNetClassifier

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dataloader = BrainDataset(csv_file = 'Healthcare_AI_Datasets/Brain_MRI/data_mask.csv', root_dir = 'Healthcare_AI_Datasets/Brain_MRI/')
train_loader, val_loader, test_loader = dataloader.get_data_loaders()

model_seg = ResUNet()  
model_seg.load_state_dict(torch.load('checkpoints/resunet_model.pth'))  
model_seg.to(device)
optimizer = optim.Adam(model_seg.parameters())
model_seg.eval()  


model = ResNetClassifier()  
model.load_state_dict(torch.load('checkpoints/classifier-resnet-model.pth')) 
model.to(device)
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters())  
model.eval()  


def prediction(test_loader, model, model_seg, device):
    # Store the results
    images_list = []
    mask_list = []
    has_mask_list = []

    # Set the models to evaluation mode
    model.eval()
    model_seg.eval()

    # No need to compute gradients
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)

            # Classification model prediction (is there a tumor or not)
            is_defect = model(images)

            # Loop through each image in the batch
            for i in range(images.size(0)):
                img = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert tensor to numpy (H, W, C)
                images_list.append(img)  # Store original image

                # If the classification model predicts no tumor
                if torch.argmax(is_defect[i]) == 0:  # No tumor
                    has_mask_list.append(0)
                    mask_list.append(None)  # No mask available
                    continue

                # If tumor is detected, proceed with segmentation
                img = images[i].unsqueeze(0)  # Add batch dimension
                img = (img - img.mean()) / img.std()  # Standardize

                # Predict mask using the segmentation model
                predicted_mask = model_seg(img).cpu().numpy()

                # Check if the predicted mask has any tumor regions
                if predicted_mask.round().astype(int).sum() == 0:
                    has_mask_list.append(0)
                    mask_list.append(None)  # No mask available
                else:
                    has_mask_list.append(1)
                    mask_list.append(predicted_mask.squeeze())  # Store the predicted mask

    return images_list, mask_list, has_mask_list


def save_image_mask_pairs_with_masks(images, masks, has_masks, save_path):
    # Filter the images and masks where has_mask is 1
    filtered_images = [img for img, has_mask in zip(images, has_masks) if has_mask == 1]
    filtered_masks = [mask for mask, has_mask in zip(masks, has_masks) if has_mask == 1]

    # Limit the number of displayed pairs to 3, if necessary
    pairs_to_display = min(3, len(filtered_images))

    if pairs_to_display == 0:
        print("No images with masks available.")
        return

    # Set up a figure to hold the selected image-mask pairs
    fig, axes = plt.subplots(2, pairs_to_display, figsize=(15, 10))  # 2 rows, `pairs_to_display` columns

    for i in range(pairs_to_display):  # Only display pairs that have masks
        # Original image
        img = filtered_images[i]

        # Display the image in the first row
        axes[0, i].imshow(img)
        axes[0, i].axis('off')  # Hide the axis for a cleaner display
        axes[0, i].set_title(f"Image {i+1}")

        # Display the corresponding mask in the second row
        mask = filtered_masks[i]
        axes[1, i].imshow(mask, cmap='gray')  # Display the mask in grayscale
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Mask {i+1}")

    # Save the figure as an image
    plt.savefig(save_path)
    plt.close()



image_id, mask, has_mask = prediction(test_loader, model, model_seg, device)
save_image_mask_pairs_with_masks(image_id, mask, has_mask, "image_mask_pairs.png")
