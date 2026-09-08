import cv2
import numpy as np

# YUV conversion constants
RGB2YUV_mat = np.array([
    [0.25678824, -0.14822353, 0.43921569],
    [0.50412941, -0.29099216, -0.36778824],
    [0.09790588, 0.43921569, -0.07142745]
], dtype=np.float32)

YUV2RGB_mat = np.array([
    [1.16438355, 1.16438355, 1.16438355],
    [0.0, -0.3917616, 2.01723105],
    [1.59602715, -0.81296805, 0.0]
], dtype=np.float32)

RGB2YUV_offset = np.array([16, 128, 128], dtype=np.float32)



def resize_infer_result_to_original(
    infer_result: np.ndarray,
    original_size: tuple[int, int],
    model_input_size: tuple[int, int]
) -> np.ndarray:
    """
    Resize and crop the super-resolution (or padded) inference result to match original image size.

    Args:
        infer_result (np.ndarray): Inference result image (H, W, C) with padding.
        original_size (tuple[int, int]): Original image size as (H_orig, W_orig).
        model_input_size (tuple[int, int]): Model input size as (H_model, W_model).

    Returns:
        np.ndarray: Resized and cropped image matching the original size.
    """
    orig_h, orig_w = original_size
    model_h, model_w = model_input_size

    # Calculate the scale and resized shape without padding
    scale = min(model_w / orig_w, model_h / orig_h)
    resized_h = int(orig_h * scale)
    resized_w = int(orig_w * scale)

    # Offsets due to padding
    x_offset = (model_w - resized_w) // 2
    y_offset = (model_h - resized_h) // 2

    # Crop only the region corresponding to the scaled image
    cropped = infer_result[y_offset:y_offset + resized_h, x_offset:x_offset + resized_w]

    # Resize to original image size
    result = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    return result


def inference_result_handler(original_frame: np.ndarray, infer_result: np.ndarray, model_height, model_width, utils=None) -> np.ndarray:
    """
    Processes a single super-resolution inference result and returns a side-by-side comparison.

    The SR result is kept at its native upscaled resolution (2× for ESRGAN, 4× for ESPCN).
    The original frame is upscaled to match for side-by-side visual comparison.

    Args:
        original_frame (np.ndarray): Original input image (H, W, 3).
        infer_result (np.ndarray): Raw model output (H', W', C).
        model_height (int): Model input height.
        model_width (int): Model input width.
        utils (SuperResolutionUtils, optional): For model-specific post-processing.

    Returns:
        np.ndarray: Side-by-side image [original_upscaled | SR_output].
    """
    # Apply model-specific post-processing (float→uint8, YUV→RGB, etc.)
    if utils is not None:
        infer_result = utils.post_process(infer_result, original_frame)

    # SR output is at upscaled resolution — resize original to match for comparison
    sr_h, sr_w = infer_result.shape[:2]
    original_resized = cv2.resize(original_frame, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)

    return np.hstack((original_resized, infer_result))

class SuperResolutionUtils:
    """
    Base class for super-resolution utility functions.

    Methods:
        pre_process(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
            Preprocesses an image for the super-resolution model.

        post_process(infer_result: np.ndarray, input_image: np.ndarray) -> np.ndarray:
            Post-processes the model's output into a final image.
    """
    def pre_process(self, image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
        raise NotImplementedError

    def post_process(self, infer_result: np.ndarray, input_image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SrganUtils(SuperResolutionUtils):
    """
    Utility class for SRGAN-specific preprocessing and postprocessing.

    Methods:
        pre_process(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
            Resizes the input image to the model dimensions.

        post_process(infer_result: np.ndarray, input_image: np.ndarray) -> np.ndarray:
            Converts the inference result back into an image.
    """
    def pre_process(self, image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
        image = cv2.resize(image, (model_w, model_h), interpolation=cv2.INTER_CUBIC)
        return image

    def post_process(self, infer_result: np.ndarray, input_image: np.ndarray) -> np.ndarray:
        if infer_result.dtype != np.uint8:
            infer_result = (infer_result * 255.0).clip(0, 255).astype(np.uint8)
        return infer_result


class Espcnx4Utils(SuperResolutionUtils):
    """
    Utility class for ESPCNx4-specific preprocessing and postprocessing.

    Attributes:
        model_w (int): Model width for postprocessing.
        model_h (int): Model height for postprocessing.

    Methods:
        pre_process(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
            Converts an image to YUV format and extracts the Y channel.

        post_process(infer_result: np.ndarray, input_image: np.ndarray) -> np.ndarray:
            Combines the SR Y-channel with resized UV channels and converts back to RGB.
    """

    def __init__(self):
        self.model_w = None
        self.model_h = None

    def pre_process(self, image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
        # Save the model width and height for post-processing
        self.model_w = model_w
        self.model_h = model_h

        # RGB --> YUV
        image = cv2.resize(image, (model_w, model_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        yuv = (image @ RGB2YUV_mat.T + RGB2YUV_offset) / 255.0
        y_channel = cv2.split(yuv)[0]  # shape (H, W)
        return y_channel[..., np.newaxis]  # shape (H, W, 1)

    def post_process(self, infer_result: np.ndarray, input_image: np.ndarray) -> np.ndarray:
        # Resize input to model dimensions and convert to float32
        input_image = cv2.resize(input_image, (self.model_w, self.model_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        # Convert input RGB --> YUV
        img_yuv = input_image @ RGB2YUV_mat.T + RGB2YUV_offset

        # Resize UV channels to match SR Y
        uv_resized = cv2.resize(img_yuv, (infer_result.shape[1], infer_result.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Prepare final YUV image
        y = infer_result * 255.0 if infer_result.dtype != np.uint8 else infer_result
        y = y.astype(np.float32)
        yuv_combined = np.concatenate([y, uv_resized[..., 1:]], axis=2)

        # YUV --> RGB
        rgb = (yuv_combined - RGB2YUV_offset) @ YUV2RGB_mat.T
        return np.clip(rgb, 0, 255).astype(np.uint8)


