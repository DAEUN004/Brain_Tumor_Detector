import torch
import numpy as np

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
