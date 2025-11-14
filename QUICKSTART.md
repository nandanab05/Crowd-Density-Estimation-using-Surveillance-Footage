# Quick Start Guide

Get started with crowd density estimation in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Test the System

Run the test script to verify everything is working:

```bash
python test_system.py
```

You should see all tests passing.

## Step 3: Prepare Your Data

### Option A: Use Your Own Video

1. Place your surveillance video in a directory (e.g., `my_video.mp4`)

2. Run the estimation:
```bash
python crowd_density_estimation.py --input my_video.mp4 --output output.mp4
```

### Option B: Set Up Sample Dataset Structure

```bash
python download_dataset.py --dataset sample
```

Then add your videos to `datasets/sample/videos/`

## Step 4: Process Your First Video

```bash
# Basic usage
python crowd_density_estimation.py --input datasets/sample/videos/your_video.mp4 --output result.mp4

# With GPU (if available)
python crowd_density_estimation.py --input your_video.mp4 --output result.mp4 --device cuda

# Process every 5th frame for faster processing
python crowd_density_estimation.py --input your_video.mp4 --output result.mp4 --frame-skip 5
```

## Step 5: Process Images

```bash
python crowd_density_estimation.py --input image.jpg --output result.jpg
```

## Step 6: Try Examples

```bash
# Example 1: Process a single image
python example_usage.py --example 1

# Example 2: Process a video
python example_usage.py --example 2

# Example 3: Real-time processing (requires camera)
python example_usage.py --example 3

# Example 4: Batch process multiple videos
python example_usage.py --example 4
```

## Common Use Cases

### Real-time Monitoring

```python
from crowd_density_estimation import CrowdDensityEstimator
import cv2

estimator = CrowdDensityEstimator(device='cpu')
cap = cv2.VideoCapture(0)  # Or RTSP URL

while True:
    ret, frame = cap.read()
    count, density = estimator.estimate_density(frame)
    vis = estimator.visualize_density(frame, density, count)
    cv2.imshow('Crowd Density', vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Batch Processing

```python
from crowd_density_estimation import CrowdDensityEstimator
import os

estimator = CrowdDensityEstimator(device='cpu')

for video_file in os.listdir('videos/'):
    if video_file.endswith('.mp4'):
        stats = estimator.process_video(
            f'videos/{video_file}',
            output_path=f'output/{video_file}',
            show_preview=False
        )
        print(f"{video_file}: {stats['avg_count']:.1f} people")
```

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Use CPU or reduce frame resolution
```bash
python crowd_density_estimation.py --input video.mp4 --device cpu
```

### Issue: "Video codec not supported"
**Solution**: Convert video to MP4 format
```bash
# Using ffmpeg (if installed)
ffmpeg -i input.avi -c:v libx264 output.mp4
```

### Issue: Processing is too slow
**Solution**: Use frame skipping
```bash
python crowd_density_estimation.py --input video.mp4 --frame-skip 10
```

## Next Steps

1. **Train on your data**: See `train_model.py` for training script
2. **Customize model**: Modify `CrowdDensityNet` in `crowd_density_estimation.py`
3. **Add features**: Extend the codebase with your requirements

## Need Help?

- Check `README.md` for detailed documentation
- Run `python test_system.py` to diagnose issues
- Review `example_usage.py` for code examples

Happy crowd counting! 🎉

