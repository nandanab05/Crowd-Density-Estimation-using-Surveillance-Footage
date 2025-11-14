"""
Example usage scripts for crowd density estimation.
"""

import cv2
import os
from crowd_density_estimation import CrowdDensityEstimator


def example_image_processing():
    """Example: Process a single image."""
    print("="*60)
    print("Example 1: Processing a Single Image")
    print("="*60)
    
    # Initialize estimator
    estimator = CrowdDensityEstimator(device='cpu')
    
    # Process image (replace with your image path)
    image_path = "datasets/sample/images/your_image.jpg"
    
    if os.path.exists(image_path):
        count, density_map = estimator.process_image(
            image_path, 
            output_path="output_density.jpg"
        )
        print(f"Estimated crowd count: {count:.2f}")
    else:
        print(f"Image not found: {image_path}")
        print("Please add an image to process.")


def example_video_processing():
    """Example: Process a video file."""
    print("\n" + "="*60)
    print("Example 2: Processing a Video")
    print("="*60)
    
    # Initialize estimator
    estimator = CrowdDensityEstimator(device='cpu')
    
    # Process video (replace with your video path)
    video_path = "datasets/sample/videos/your_video.mp4"
    
    if os.path.exists(video_path):
        stats = estimator.process_video(
            video_path,
            output_path="output_density_video.mp4",
            show_preview=True,
            frame_skip=5  # Process every 5th frame for faster processing
        )
        print(f"\nVideo processing statistics:")
        print(f"  Average count: {stats['avg_count']:.2f}")
        print(f"  Max count: {stats['max_count']:.2f}")
        print(f"  Min count: {stats['min_count']:.2f}")
    else:
        print(f"Video not found: {video_path}")
        print("Please add a video to process.")


def example_realtime_processing():
    """Example: Process video from webcam or RTSP stream."""
    print("\n" + "="*60)
    print("Example 3: Real-time Processing from Camera")
    print("="*60)
    
    # Initialize estimator
    estimator = CrowdDensityEstimator(device='cpu')
    
    # Open camera (0 for default webcam, or RTSP URL for IP camera)
    # For RTSP: cap = cv2.VideoCapture("rtsp://username:password@ip:port/stream")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open camera")
        return
    
    print("Press 'q' to quit")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 10th frame for performance
        if frame_count % 10 == 0:
            count, density_map = estimator.estimate_density(frame)
            vis_frame = estimator.visualize_density(frame, density_map, count)
        else:
            vis_frame = frame
            cv2.putText(vis_frame, "Processing...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Real-time Crowd Density', vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()


def example_batch_processing():
    """Example: Process multiple videos in a directory."""
    print("\n" + "="*60)
    print("Example 4: Batch Processing Multiple Videos")
    print("="*60)
    
    # Initialize estimator
    estimator = CrowdDensityEstimator(device='cpu')
    
    # Directory containing videos
    video_dir = "datasets/sample/videos"
    output_dir = "output_videos"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(video_dir):
        print(f"Directory not found: {video_dir}")
        return
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in os.listdir(video_dir) 
                  if any(f.lower().endswith(ext) for ext in video_extensions)]
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video(s) to process")
    
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        output_path = os.path.join(output_dir, f"density_{video_file}")
        
        print(f"\nProcessing: {video_file}")
        try:
            stats = estimator.process_video(
                video_path,
                output_path=output_path,
                show_preview=False,
                frame_skip=5
            )
            print(f"  Average count: {stats['avg_count']:.2f}")
        except Exception as e:
            print(f"  Error processing {video_file}: {e}")
    
    print(f"\nBatch processing complete! Outputs saved to: {output_dir}")


def main():
    """Run examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Example usage of crowd density estimation')
    parser.add_argument('--example', type=int, choices=[1, 2, 3, 4], default=2,
                       help='Example to run (1=image, 2=video, 3=realtime, 4=batch)')
    
    args = parser.parse_args()
    
    if args.example == 1:
        example_image_processing()
    elif args.example == 2:
        example_video_processing()
    elif args.example == 3:
        example_realtime_processing()
    elif args.example == 4:
        example_batch_processing()


if __name__ == '__main__':
    main()

