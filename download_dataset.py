"""
Script to download and prepare datasets for crowd density estimation.
Supports multiple popular datasets.
"""

import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Optional


def download_file(url: str, output_path: str, description: str = ""):
    """Download a file with progress bar."""
    print(f"Downloading {description}...")
    print(f"URL: {url}")
    print(f"Output: {output_path}")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    
    urllib.request.urlretrieve(url, output_path, show_progress)
    print("\nDownload complete!")


def extract_zip(zip_path: str, extract_to: str):
    """Extract a zip file."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")


def extract_tar(tar_path: str, extract_to: str):
    """Extract a tar file."""
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r:*') as tar_ref:
        tar_ref.extractall(extract_to)
    print("Extraction complete!")


def setup_shanghaitech_dataset(dataset_dir: str = "datasets"):
    """
    Setup instructions for ShanghaiTech dataset.
    Note: This dataset requires manual download due to licensing.
    """
    print("\n" + "="*60)
    print("ShanghaiTech Dataset Setup")
    print("="*60)
    print("\nThe ShanghaiTech dataset is available at:")
    print("https://github.com/desenzhou/ShanghaiTechDataset")
    print("\nPlease download the dataset manually and extract it to:")
    print(f"{os.path.abspath(dataset_dir)}/ShanghaiTech")
    print("\nThe dataset should have the following structure:")
    print("ShanghaiTech/")
    print("  ├── part_A/")
    print("  │   ├── train_data/")
    print("  │   └── test_data/")
    print("  └── part_B/")
    print("      ├── train_data/")
    print("      └── test_data/")


def setup_ucf_cc_50_dataset(dataset_dir: str = "datasets"):
    """
    Setup instructions for UCF_CC_50 dataset.
    """
    print("\n" + "="*60)
    print("UCF_CC_50 Dataset Setup")
    print("="*60)
    print("\nThe UCF_CC_50 dataset is available at:")
    print("https://www.crcv.ucf.edu/data/ucf-cc-50/")
    print("\nPlease download the dataset manually and extract it to:")
    print(f"{os.path.abspath(dataset_dir)}/UCF_CC_50")


def create_sample_dataset(output_dir: str = "datasets/sample"):
    """
    Create a sample dataset structure for testing.
    This creates directories and a README for organizing your own data.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
    
    readme_content = """# Sample Dataset Structure

This directory is for organizing your own crowd density estimation dataset.

## Directory Structure:
- `images/`: Place your training/test images here
- `annotations/`: Place ground truth density maps or point annotations here
- `videos/`: Place your surveillance video footage here

## Dataset Format:

### Images
- Supported formats: JPG, PNG
- Recommended resolution: 640x480 or higher
- Multiple images can be organized in subdirectories

### Annotations
- Ground truth density maps (numpy arrays or images)
- Or point annotations (CSV format: x, y coordinates)

### Videos
- Supported formats: MP4, AVI, MOV
- Can be used for inference or frame extraction

## Usage:
1. Place your surveillance footage in the `videos/` directory
2. Extract frames using: `python video_utils.py --extract-frames`
3. Annotate images if training a new model
4. Use the images for training or testing
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\nSample dataset structure created at: {os.path.abspath(output_dir)}")
    print("Please add your own images and videos to the respective directories.")


def download_sample_video(output_dir: str = "datasets/sample/videos"):
    """
    Download a sample video for testing (if available).
    Note: In practice, you'll need to provide your own surveillance footage.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Sample Video Download")
    print("="*60)
    print("\nFor testing purposes, you can:")
    print("1. Use your own surveillance footage")
    print("2. Download sample crowd videos from:")
    print("   - YouTube (with proper permissions)")
    print("   - Public datasets")
    print("   - Your own security camera recordings")
    print(f"\nPlace videos in: {os.path.abspath(output_dir)}")


def main():
    """Main function to set up datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup datasets for crowd density estimation')
    parser.add_argument('--dataset', type=str, 
                       choices=['shanghaitech', 'ucf_cc_50', 'sample', 'all'],
                       default='sample',
                       help='Dataset to set up')
    parser.add_argument('--output-dir', type=str, default='datasets',
                       help='Output directory for datasets')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dataset == 'shanghaitech' or args.dataset == 'all':
        setup_shanghaitech_dataset(args.output_dir)
    
    if args.dataset == 'ucf_cc_50' or args.dataset == 'all':
        setup_ucf_cc_50_dataset(args.output_dir)
    
    if args.dataset == 'sample' or args.dataset == 'all':
        create_sample_dataset(os.path.join(args.output_dir, 'sample'))
        download_sample_video(os.path.join(args.output_dir, 'sample', 'videos'))
    
    print("\n" + "="*60)
    print("Dataset Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Add your surveillance footage to the dataset directories")
    print("2. Run: python crowd_density_estimation.py --input <video_path>")
    print("3. For training, prepare annotations and use the training scripts")


if __name__ == '__main__':
    main()

