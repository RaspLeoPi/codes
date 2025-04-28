import numpy as np
from scipy.ndimage import rotate, shift, zoom

def random_rotation(image, max_angle=15):
    """
    Apply random rotation to MNIST image (28x28)
    Args:
        image: 1D array (784,) or 2D array (28,28)
        max_angle: maximum rotation angle in degrees (+/-)
    Returns:
        Rotated image with same dimensions
    """
    angle = np.random.uniform(-max_angle, max_angle)
    if image.ndim == 1:
        image = image.reshape(28, 28)
    rotated = rotate(image, angle, reshape=False, mode='nearest')
    return rotated.reshape(-1)  # Flatten back to 1D if needed

def random_shift(image, max_shift=2):
    """
    Apply random translation to MNIST image
    Args:
        image: 1D array (784,) or 2D array (28,28)
        max_shift: maximum pixels to shift (+/-)
    Returns:
        Shifted image with same dimensions
    """
    dx, dy = np.random.randint(-max_shift, max_shift+1, size=2)
    if image.ndim == 1:
        image = image.reshape(28, 28)
    shifted = shift(image, (dy, dx), mode='constant', cval=0)
    return shifted.reshape(-1)

def random_zoom(image, min_scale=0.9, max_scale=1.1):
    """
    Apply random scaling to MNIST image
    Args:
        image: 1D array (784,) or 2D array (28,28)
        min_scale: minimum zoom factor
        max_scale: maximum zoom factor
    Returns:
        Zoomed image with same dimensions (cropped/padded as needed)
    """
    scale = np.random.uniform(min_scale, max_scale)
    if image.ndim == 1:
        image = image.reshape(28, 28)
    
    # Calculate new dimensions
    h, w = image.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Zoom and then crop/pad to original size
    zoomed = zoom(image, scale, mode='nearest')
    if scale > 1:  # Crop center
        start_h = (zoomed.shape[0] - h) // 2
        start_w = (zoomed.shape[1] - w) // 2
        zoomed = zoomed[start_h:start_h+h, start_w:start_w+w]
    else:  # Pad with zeros
        pad_h = (h - zoomed.shape[0]) / 2
        pad_w = (w - zoomed.shape[1]) / 2
        zoomed = np.pad(zoomed, ((np.floor(pad_h).astype(int), np.ceil(pad_h).astype(int)),
                                (np.floor(pad_w).astype(int), np.ceil(pad_w).astype(int))), mode='constant')
    
    return zoomed.reshape(-1)

def augment_image(image):
    """
    Apply random augmentation pipeline to MNIST image
    Args:
        image: 1D array (784,) or 2D array (28,28)
    Returns:
        Augmented image with same dimensions
    """
    # Apply transformations with 50% probability each
    if np.random.rand() > 0.5:
        image = random_rotation(image)
    if np.random.rand() > 0.5:
        image = random_shift(image)
    if np.random.rand() > 0.5:
        image = random_zoom(image)
    return image
