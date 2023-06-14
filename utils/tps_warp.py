import numpy as np
import torch
from PIL import Image, ImageOps
from skimage.transform import PiecewiseAffineTransform, warp
from utils.HumanPartsSegmentation.simple_extractor import extract_parts

def tps_warp(M_mask_path, Target_c_path):
    def create_tps_warp_model(M_mask, c_mask):
        # Convert masks to numpy arrays
        M_np = M_mask.cpu().numpy()
        c_np = c_mask.cpu().numpy()

        # Get the coordinates of non-zero values in M_mask
        M_coords = np.argwhere(M_np > 0)

        # Get the corresponding coordinates in c_mask
        c_coords = M_coords + (c_np.shape[0] - M_np.shape[0], c_np.shape[1] - M_np.shape[1])

        # Create a Thin Plate Spline (TPS) warp model
        tps_model = PiecewiseAffineTransform()
        tps_model.estimate(M_coords, c_coords)

        return tps_model

    def threshold_mask(mask, threshold):
        # Apply thresholding to the mask
        thresholded_mask = mask > threshold

        return thresholded_mask

    def warp_target(c, tps_model, output_shape):
        # Warp the target c using the TPS model
        warped_c = warp(c, tps_model.inverse, output_shape=output_shape)

        return warped_c

    M_mask = ImageOps.grayscale(Image.open(M_mask_path))
    target_c = Image.open(Target_c_path)
    c_mask = extract_parts(Target_c_path,)

    tps_model = create_tps_warp_model(M_mask, c_mask)
    output_shape = M_mask.shape
    warped_target_c = warp(target_c, tps_model, output_shape)

    return warped_target_c
