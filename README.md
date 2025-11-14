# Crowd Density Estimation from Surveillance Footage

A Python-based system for estimating crowd density in surveillance videos using deep learning. This implementation uses a CNN-based architecture inspired by CSRNet for accurate crowd counting and density map generation.

## Features

- **Real-time Processing**: Process video streams in real-time
- **Batch Processing**: Process multiple videos efficiently
- **Density Visualization**: Visualize crowd density with heat maps
- **Multiple Input Formats**: Supports images, videos, and camera streams
- **Flexible Architecture**: Easy to extend and customize

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone or download this repository:
```bash
cd crowd
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up datasets:
```bash
python download_dataset.py --dataset sample
```

## Quick Start

### Process a Video

```bash
python crowd_density_estimation.py --input path/to/your/video.mp4 --output output.mp4
```

### Process an Image

```bash
python crowd_density_estimation.py --input path/to/your/image.jpg --output output.jpg
```

### Process with GPU (if available)

```bash
python crowd_density_estimation.py --input video.mp4 --device cuda --output output.mp4
```

## Usage Examples

### Example 1: Process a Single Image

```python
from crowd_density_estimation import CrowdDensityEstimator

estimator = CrowdDensityEstimator(device='cpu')
count, density_map = estimator.process_image('input.jpg', 'output.jpg')
print(f"Estimated count: {count:.2f}")
```

### Example 2: Process a Video

```python
from crowd_density_estimation import CrowdDensityEstimator

estimator = CrowdDensityEstimator(device='cpu')
stats = estimator.process_video(
    'input.mp4',
    output_path='output.mp4',
    show_preview=True,
    frame_skip=5  # Process every 5th frame
)
print(f"Average count: {stats['avg_count']:.2f}")
```

### Example 3: Real-time Processing

```python
from crowd_density_estimation import CrowdDensityEstimator
import cv2

estimator = CrowdDensityEstimator(device='cpu')
cap = cv2.VideoCapture(0)  # Or RTSP URL for IP camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    count, density_map = estimator.estimate_density(frame)
    vis_frame = estimator.visualize_density(frame, density_map, count)
    cv2.imshow('Crowd Density', vis_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## Command Line Options

```
usage: crowd_density_estimation.py [-h] --input INPUT [--output OUTPUT]
                                   [--model MODEL] [--device {cpu,cuda}]
                                   [--no-preview] [--frame-skip FRAME_SKIP]

optional arguments:
  --input INPUT        Path to input video or image
  --output OUTPUT      Path to output video/image
  --model MODEL        Path to pre-trained model weights
  --device {cpu,cuda} Device to use (cpu or cuda)
  --no-preview         Disable preview window
  --frame-skip FRAME_SKIP
                       Process every Nth frame
```

## Dataset Setup

### Using Your Own Data

1. Create a dataset directory:
```bash
python download_dataset.py --dataset sample
```

2. Add your surveillance footage:
   - Place videos in `datasets/sample/videos/`
   - Place images in `datasets/sample/images/`

3. Process your data:
```bash
python crowd_density_estimation.py --input datasets/sample/videos/your_video.mp4
```

### Public Datasets

For training models, you can use these popular datasets:

1. **ShanghaiTech Dataset**
   - Download from: https://github.com/desenzhou/ShanghaiTechDataset
   - Setup: `python download_dataset.py --dataset shanghaitech`

2. **UCF_CC_50 Dataset**
   - Download from: https://www.crcv.ucf.edu/data/ucf-cc-50/
   - Setup: `python download_dataset.py --dataset ucf_cc_50`

3. **WorldExpo'10 Dataset**
   - Available for research purposes
   - Contact dataset authors for access

## Model Architecture

The implementation uses a CNN-based architecture with:

- **Front-end**: VGG-like feature extraction layers
- **Back-end**: Dilated convolutions for density map generation
- **Output**: Single-channel density map

The model can be trained on crowd counting datasets to improve accuracy.

## Performance Tips

1. **Use GPU**: Set `--device cuda` for faster processing
2. **Frame Skipping**: Use `--frame-skip` to process fewer frames
3. **Resolution**: Lower resolution videos process faster
4. **Batch Processing**: Process multiple videos offline

## Output Format

The system generates:
- **Density Map**: Heat map visualization of crowd density
- **Count**: Estimated number of people in the frame
- **Statistics**: For videos, provides average, min, max counts

## File Structure

```
crowd/
├── crowd_density_estimation.py  # Main estimation script
├── video_utils.py               # Video processing utilities
├── download_dataset.py          # Dataset setup script
├── example_usage.py             # Usage examples
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce frame resolution or use CPU
2. **Video codec errors**: Install codec packages or convert video format
3. **Slow processing**: Use frame skipping or reduce resolution
4. **Model not found**: The model will initialize without pre-trained weights

## Training Your Own Model

To train a model on your own data:

1. Prepare annotated dataset (images + density maps or point annotations)
2. Implement training script (see model architecture in `crowd_density_estimation.py`)
3. Train using PyTorch
4. Save model weights and use with `--model` parameter

## License

This code is provided for educational and research purposes. Please ensure you have proper permissions when using surveillance footage.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## References

- CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes
- MCNN: Multi-column Convolutional Neural Network for Crowd Counting
- ShanghaiTech Dataset: https://github.com/desenzhou/ShanghaiTechDataset

## Support

For questions or issues, please open an issue on the repository or refer to the example usage scripts.

