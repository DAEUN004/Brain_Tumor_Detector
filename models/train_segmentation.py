import sys
import os
import torch.optim as optim
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import BrainDataset
from models.loss import focal_tversky, tversky_loss, tversky
from models.resunet import ResUNet
from models.train import EarlyStopping

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

dataloader = BrainDataset(csv_file = 'Healthcare_AI_Datasets/Brain_MRI/data_mask.csv', root_dir = 'Healthcare_AI_Datasets/Brain_MRI/', segmentation=True)
train_loader, val_loader, test_loader = dataloader.get_data_loaders()

model_seg = ResUNet()
optimizer = optim.Adam(model_seg.parameters(), lr=0.05, eps=0.1)
loss_fn = focal_tversky

num_epochs = 20
patience = 20
best_val_loss = float("inf")
early_stopping_counter = 0

model_seg.to(device)

for epoch in range(num_epochs):
    model_seg.train()  # Set model to training mode
    running_loss = 0.0
    total = 0

    # Training loop
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model_seg(images)
        loss = loss_fn(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += masks.size(0)

    train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")

    # Validation step
    model_seg.eval()  # Set model to evaluation mode
    val_loss = 0.0
    total = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model_seg(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")

    # Save the best model checkpoint
    is_best = val_loss < best_val_loss
    best_val_loss = min(val_loss, best_val_loss)
    if is_best:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_seg.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, "best_model_checkpoint.pth")
        print(f"Saved best model at epoch {epoch+1}")

    # EarlyStopping(val_loss, model_seg)
    # if EarlyStopping.early_stop:
    #     print("Early stopping triggered.")
    #     break

# Save the final model
torch.save(model_seg.state_dict(), 'resunet_model.pth')