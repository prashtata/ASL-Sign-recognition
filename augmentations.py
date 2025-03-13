import numpy as np
import torch

def jitter_keypoints(data, noise_level=0.05):
    """Add small Gaussian noise to keypoints."""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def scale_keypoints(data, scale_range=(0.8, 1.2)):
    """Randomly scale the keypoints."""
    scale = np.random.uniform(*scale_range)
    return data * scale

def time_warp_keypoints(data, sigma=0.2):
    """Randomly stretch or compress parts of the sequence."""
    time_steps = np.arange(data.shape[0])
    distortions = np.random.normal(0, sigma, size=data.shape[0])
    distorted_time_steps = np.clip(time_steps + distortions, 0, data.shape[0] - 1).astype(int)
    return data[distorted_time_steps]

def augment_data(data):
    """Apply multiple augmentations to increase dataset size by 4x."""
    augmented_data = []
    
    for sample in data:
        # Original
        augmented_data.append(sample)
        
        # Augmented versions
        augmented_data.append(jitter_keypoints(sample))
        augmented_data.append(scale_keypoints(sample))
        augmented_data.append(time_warp_keypoints(sample))
    
    return np.array(augmented_data)