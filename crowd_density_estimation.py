import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from typing import Tuple, Optional


class CrowdDensityNet(nn.Module):
    def __init__(self, load_weights: bool = False, weights_path: Optional[str] = None):
        super(CrowdDensityNet, self).__init__()
        
        # Front-end feature extraction (VGG-like)
        self.frontend_feat = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Back-end density map generation
        self.backend_density = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        
        # Output layer
        self.output_layer = nn.Conv2d(64, 1, 1)
        
        if load_weights and weights_path and os.path.exists(weights_path):
            self.load_state_dict(torch.load(weights_path, map_location='cpu'))
            print(f"Loaded weights from {weights_path}")
    
    def forward(self, x):
        x = self.frontend_feat(x)
        x = self.backend_density(x)
        x = self.output_layer(x)
        return x


class CrowdDensityEstimator:
    """
    Main class for crowd density estimation from video frames.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize the crowd density estimator.
        
        Args:
            model_path: Path to pre-trained model weights (optional)
            device: 'cpu' or 'cuda'
        """
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = CrowdDensityNet(load_weights=model_path is not None, 
                                     weights_path=model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model initialized on device: {self.device}")
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess a video frame for the model.
        
        Args:
            frame: Input frame as numpy array (BGR format from OpenCV)
        
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (can be adjusted)
        frame_resized = cv2.resize(frame_rgb, (640, 480))
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_resized)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def estimate_density(self, frame: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Estimate crowd density for a single frame.
        
        Args:
            frame: Input frame as numpy array
        
        Returns:
            Tuple of (count, density_map)
        """
        with torch.no_grad():
            # Preprocess frame
            input_tensor = self.preprocess_frame(frame)
            
            # Forward pass
            density_map = self.model(input_tensor)
            
            # Calculate total count
            count = density_map.sum().item()
            
            # Convert density map to numpy for visualization
            density_map_np = density_map.squeeze().cpu().numpy()
            
            # Resize density map to original frame size
            h, w = frame.shape[:2]
            density_map_resized = cv2.resize(density_map_np, (w, h))
            
            return count, density_map_resized
    
    def process_video(self, video_path: str, output_path: Optional[str] = None, 
                     show_preview: bool = True, frame_skip: int = 1) -> dict:
        """
        Process a video file and estimate crowd density for each frame.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional)
            show_preview: Whether to show preview window
            frame_skip: Process every Nth frame (1 = all frames)
        
        Returns:
            Dictionary with statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        counts = []
        
        print("Processing video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_count % frame_skip == 0:
                # Estimate density
                count, density_map = self.estimate_density(frame)
                counts.append(count)
                
                # Visualize results
                vis_frame = self.visualize_density(frame, density_map, count)
                
                # Write to output video
                if writer:
                    writer.write(vis_frame)
                
                # Show preview
                if show_preview:
                    cv2.imshow('Crowd Density Estimation', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} frames...")
            
            frame_count += 1
        
        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        # Calculate statistics
        stats = {
            'total_frames': total_frames,
            'processed_frames': processed_count,
            'avg_count': np.mean(counts) if counts else 0,
            'max_count': np.max(counts) if counts else 0,
            'min_count': np.min(counts) if counts else 0,
            'std_count': np.std(counts) if counts else 0
        }
        
        print("\nProcessing complete!")
        print(f"Average crowd count: {stats['avg_count']:.2f}")
        print(f"Max crowd count: {stats['max_count']:.2f}")
        print(f"Min crowd count: {stats['min_count']:.2f}")
        
        return stats
    
    def visualize_density(self, frame: np.ndarray, density_map: np.ndarray, 
                         count: float) -> np.ndarray:
        """
        Visualize density map on the frame.
        
        Args:
            frame: Original frame
            density_map: Density map from model
            count: Estimated count
        
        Returns:
            Visualization frame
        """
        # Normalize density map for visualization
        density_normalized = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
        density_colored = cv2.applyColorMap(density_normalized.astype(np.uint8), 
                                           cv2.COLORMAP_JET)
        
        # Blend with original frame
        overlay = cv2.addWeighted(frame, 0.6, density_colored, 0.4, 0)
        
        # Add text with count
        text = f"Count: {count:.1f}"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Add density level indicator
        density_level = "Low" if count < 10 else "Medium" if count < 50 else "High"
        cv2.putText(overlay, f"Density: {density_level}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        return overlay
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Tuple[float, np.ndarray]:
        """
        Process a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
        
        Returns:
            Tuple of (count, density_map)
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        count, density_map = self.estimate_density(frame)
        
        if output_path:
            vis_frame = self.visualize_density(frame, density_map, count)
            cv2.imwrite(output_path, vis_frame)
            print(f"Output saved to: {output_path}")
        
        return count, density_map


def main():
    """
    Example usage of the crowd density estimator.
    """
    import argparse
    import sys
    
    # Custom error handler to suppress argparse's default error messages
    class CustomArgumentParser(argparse.ArgumentParser):
        def error(self, message):
            self.print_help()
            print(f"\nError: {message}")
            sys.exit(1)
    
    parser = CustomArgumentParser(description='Crowd Density Estimation')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input video or image')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output video/image')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to pre-trained model weights')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable preview window')
    parser.add_argument('--frame-skip', type=int, default=1,
                       help='Process every Nth frame')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Initialize estimator
    estimator = CrowdDensityEstimator(model_path=args.model, device=args.device)
    
    # Check if input is image or video
    input_ext = os.path.splitext(args.input)[1].lower()
    
    if input_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process image
        count, density_map = estimator.process_image(args.input, args.output)
        print(f"\nEstimated crowd count: {count:.2f}")
    else:
        # Process video
        stats = estimator.process_video(
            args.input, 
            args.output, 
            show_preview=not args.no_preview,
            frame_skip=args.frame_skip
        )
        print(f"\nStatistics: {stats}")


if __name__ == '__main__':
    main()

