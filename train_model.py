"""
Training script for crowd density estimation model.
This is a template that can be adapted for your specific dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

from crowd_density_estimation import CrowdDensityNet


class CrowdDataset(Dataset):
    """
    Dataset class for crowd density estimation.
    Adapt this to your specific dataset format.
    """
    
    def __init__(self, image_dir, density_dir, transform=None):
        """
        Args:
            image_dir: Directory containing input images
            density_dir: Directory containing ground truth density maps
            transform: Optional transform to apply
        """
        self.image_dir = Path(image_dir)
        self.density_dir = Path(density_dir)
        self.transform = transform
        
        # Get list of image files
        self.image_files = sorted(list(self.image_dir.glob('*.jpg')) + 
                                 list(self.image_dir.glob('*.png')))
        
        print(f"Found {len(self.image_files)} images in dataset")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to standard size
        image = cv2.resize(image, (640, 480))
        image = image.astype(np.float32) / 255.0
        
        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor and change to CHW format
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Load density map
        density_path = self.density_dir / (image_path.stem + '.npy')
        if density_path.exists():
            density_map = np.load(density_path)
        else:
            # If density map doesn't exist, create zero map
            density_map = np.zeros((480, 640), dtype=np.float32)
        
        # Resize density map to match image
        density_map = cv2.resize(density_map, (640, 480))
        density_map = density_map.astype(np.float32)
        
        # Convert to tensor
        density_map = torch.from_numpy(density_map).unsqueeze(0).float()
        
        return image, density_map


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, density_maps in pbar:
        images = images.to(device)
        density_maps = density_maps.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, density_maps)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            pred_count = outputs.sum(dim=(1, 2, 3))
            gt_count = density_maps.sum(dim=(1, 2, 3))
            mae = torch.abs(pred_count - gt_count).mean().item()
            mse = ((pred_count - gt_count) ** 2).mean().item()
        
        total_loss += loss.item()
        total_mae += mae
        total_mse += mse
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mae': f'{mae:.2f}',
            'mse': f'{mse:.2f}'
        })
    
    return total_loss / len(dataloader), total_mae / len(dataloader), total_mse / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, density_maps in pbar:
            images = images.to(device)
            density_maps = density_maps.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, density_maps)
            
            pred_count = outputs.sum(dim=(1, 2, 3))
            gt_count = density_maps.sum(dim=(1, 2, 3))
            mae = torch.abs(pred_count - gt_count).mean().item()
            mse = ((pred_count - gt_count) ** 2).mean().item()
            
            total_loss += loss.item()
            total_mae += mae
            total_mse += mse
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{mae:.2f}',
                'mse': f'{mse:.2f}'
            })
    
    return total_loss / len(dataloader), total_mae / len(dataloader), total_mse / len(dataloader)


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train crowd density estimation model')
    parser.add_argument('--train-images', type=str, required=True,
                       help='Directory containing training images')
    parser.add_argument('--train-density', type=str, required=True,
                       help='Directory containing training density maps')
    parser.add_argument('--val-images', type=str, default=None,
                       help='Directory containing validation images')
    parser.add_argument('--val-density', type=str, default=None,
                       help='Directory containing validation density maps')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = CrowdDataset(args.train_images, args.train_density)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=2)
    
    val_loader = None
    if args.val_images and args.val_density:
        val_dataset = CrowdDataset(args.val_images, args.val_density)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=2)
    
    # Create model
    model = CrowdDensityNet()
    model.to(device)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    best_val_mae = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_mae, train_mse = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        print(f"Train - Loss: {train_loss:.4f}, MAE: {train_mae:.2f}, MSE: {train_mse:.2f}")
        
        # Validate
        if val_loader:
            val_loss, val_mae, val_mse = validate(
                model, val_loader, criterion, device
            )
            print(f"Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.2f}, MSE: {val_mse:.2f}")
            
            # Save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': val_mae,
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print(f"Saved best model (MAE: {val_mae:.2f})")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))
        
        # Update learning rate
        scheduler.step()
    
    print("\nTraining complete!")
    print(f"Best validation MAE: {best_val_mae:.2f}")


if __name__ == '__main__':
    main()

