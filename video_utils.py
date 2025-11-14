"""
Utility functions for video processing and frame extraction.
"""

import cv2
import numpy as np
import os
from typing import List, Optional, Tuple


def extract_frames(video_path: str, output_dir: str, 
                  frame_interval: int = 1, max_frames: Optional[int] = None) -> List[str]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (1 = all frames)
        max_frames: Maximum number of frames to extract (None = all)
    
    Returns:
        List of paths to extracted frames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_paths = []
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1
            
            if max_frames and saved_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")
    return frame_paths


def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


def create_video_from_frames(frame_dir: str, output_path: str, fps: float = 30.0):
    """
    Create a video from a directory of frames.
    
    Args:
        frame_dir: Directory containing frames
        output_path: Path to save output video
        fps: Frames per second for output video
    """
    frame_files = sorted([f for f in os.listdir(frame_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not frame_files:
        raise ValueError(f"No image files found in {frame_dir}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            writer.write(frame)
    
    writer.release()
    print(f"Video created: {output_path}")


def resize_video(input_path: str, output_path: str, 
                target_width: Optional[int] = None, 
                target_height: Optional[int] = None,
                scale_factor: Optional[float] = None):
    """
    Resize a video file.
    
    Args:
        input_path: Path to input video
        output_path: Path to save resized video
        target_width: Target width (optional)
        target_height: Target height (optional)
        scale_factor: Scale factor (optional, e.g., 0.5 for half size)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    # Get original dimensions
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate new dimensions
    if scale_factor:
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
    elif target_width and target_height:
        new_width = target_width
        new_height = target_height
    elif target_width:
        new_height = int(orig_height * (target_width / orig_width))
        new_width = target_width
    elif target_height:
        new_width = int(orig_width * (target_height / orig_height))
        new_height = target_height
    else:
        raise ValueError("Must provide target_width, target_height, or scale_factor")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, (new_width, new_height))
        writer.write(resized_frame)
    
    cap.release()
    writer.release()
    print(f"Resized video saved to: {output_path}")


def split_video(input_path: str, output_dir: str, segment_duration: float):
    """
    Split a video into segments of specified duration.
    
    Args:
        input_path: Path to input video
        output_dir: Directory to save video segments
        segment_duration: Duration of each segment in seconds
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_segment = int(fps * segment_duration)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    segment_num = 0
    frame_count = 0
    writer = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frames_per_segment == 0:
            if writer:
                writer.release()
            
            output_path = os.path.join(output_dir, f"segment_{segment_num:04d}.mp4")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            segment_num += 1
        
        if writer:
            writer.write(frame)
        
        frame_count += 1
    
    if writer:
        writer.release()
    cap.release()
    
    print(f"Split video into {segment_num} segments in {output_dir}")

