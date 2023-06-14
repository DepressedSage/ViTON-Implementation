from PIL import Image
import cv2
import numpy as np

def downsample_body_mask(body_mask):

    image = Image.fromarray(body_mask.astype(np.uint8))
    # Resize the image to a smaller size using downsampling
    downsampled_image = image.resize((16, 16), Image.BOX)
    
    # Resize the downscaled image back to the original size using upsampling
    pixelated_image = downsampled_image.resize(image.size, Image.NEAREST)
    
    return np.asarray(pixelated_image)