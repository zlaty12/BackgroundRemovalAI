import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from model import UNet  # Assume this imports your UNet model definition
from dataset import BackgroundRemovalDataset  # Assume this imports your dataset class
from torch.cuda.amp import GradScaler, autocast
import time
import multiprocessing

def main():
    # Hyperparameters
    BATCH_SIZE = 42  # Increased batch sizepy
    LEARNING_RATE = 0.001
    EPOCHS = 100
    IMAGE_SIZE = 256
    VAL_SPLIT = 0.1  # 10% of data for validation

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # Transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Dataset and DataLoader
    full_dataset = BackgroundRemovalDataset('D:/AIEMB/ProcessedData/images', 'D:/AIEMB/ProcessedData/masks', transform=train_transform)
    
    # Split dataset into train and validation
    val_size = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = val_transform  # Use different transform for validation

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)  # Increased num_workers
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # Model
    model = UNet(n_channels=3, n_classes=1).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Gradient Scaler for mixed precision training
    scaler = GradScaler()

    # Training loop
    start_time = time.time()
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        model.train()
        train_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Mixed precision training
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            print(f'Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}')
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                print(f'Epoch [{epoch+1}/{EPOCHS}], Validation Batch [{i+1}/{len(val_loader)}], Val Loss: {loss.item():.4f}')
        
        val_loss /= len(val_loader)
        
        print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_background_removal_model.pth')
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time / 60:.2f} minutes")
    print("Training completed and best model saved.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()