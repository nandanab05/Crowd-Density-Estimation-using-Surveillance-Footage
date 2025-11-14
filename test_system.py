"""
Test script to verify the crowd density estimation system is working correctly.
"""

import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    try:
        import torch
        import torchvision
        import cv2
        import numpy as np
        from PIL import Image
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_model_creation():
    """Test if the model can be created."""
    print("\nTesting model creation...")
    try:
        from crowd_density_estimation import CrowdDensityNet
        
        model = CrowdDensityNet()
        print("✓ Model created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 480, 640)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Model forward pass successful (output shape: {output.shape})")
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False


def test_estimator_initialization():
    """Test if the estimator can be initialized."""
    print("\nTesting estimator initialization...")
    try:
        from crowd_density_estimation import CrowdDensityEstimator
        
        estimator = CrowdDensityEstimator(device='cpu')
        print("✓ Estimator initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Estimator initialization error: {e}")
        return False


def test_image_processing():
    """Test image processing with a dummy image."""
    print("\nTesting image processing...")
    try:
        from crowd_density_estimation import CrowdDensityEstimator
        
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        estimator = CrowdDensityEstimator(device='cpu')
        count, density_map = estimator.estimate_density(dummy_image)
        
        print(f"✓ Image processing successful (count: {count:.2f})")
        print(f"  Density map shape: {density_map.shape}")
        return True
    except Exception as e:
        print(f"✗ Image processing error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_utils():
    """Test video utility functions."""
    print("\nTesting video utilities...")
    try:
        from video_utils import get_video_info
        
        # This will fail if no video is available, which is expected
        print("✓ Video utilities module loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Video utilities error: {e}")
        return False


def create_test_image():
    """Create a test image for testing."""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple test image with some patterns
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate a scene
    cv2.rectangle(test_image, (50, 50), (200, 200), (100, 100, 100), -1)
    cv2.rectangle(test_image, (300, 150), (500, 350), (150, 150, 150), -1)
    cv2.circle(test_image, (400, 300), 50, (200, 200, 200), -1)
    
    test_image_path = test_dir / "test_image.jpg"
    cv2.imwrite(str(test_image_path), test_image)
    print(f"\nCreated test image: {test_image_path}")
    return test_image_path


def run_full_test():
    """Run a full test with a created test image."""
    print("\n" + "="*60)
    print("Running Full System Test")
    print("="*60)
    
    # Create test image
    test_image_path = create_test_image()
    
    try:
        from crowd_density_estimation import CrowdDensityEstimator
        
        estimator = CrowdDensityEstimator(device='cpu')
        count, density_map = estimator.process_image(
            str(test_image_path),
            output_path="test_data/test_output.jpg"
        )
        
        print(f"\n✓ Full test successful!")
        print(f"  Estimated count: {count:.2f}")
        print(f"  Output saved to: test_data/test_output.jpg")
        return True
    except Exception as e:
        print(f"\n✗ Full test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Crowd Density Estimation System Test")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Estimator Initialization", test_estimator_initialization),
        ("Image Processing", test_image_processing),
        ("Video Utilities", test_video_utils),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Run full test
    try:
        full_test_result = run_full_test()
        results.append(("Full System Test", full_test_result))
    except Exception as e:
        print(f"✗ Full system test failed: {e}")
        results.append(("Full System Test", False))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

