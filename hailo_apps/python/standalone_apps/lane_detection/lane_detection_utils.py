import sys
from multiprocessing import Process
from math import hypot
from pathlib import Path
import numpy as np
import cv2

try:
    from hailo_apps.python.core.common.hailo_logger import get_logger
except ImportError:
    core_dir = Path(__file__).resolve().parents[2] / "core"
    sys.path.insert(0, str(core_dir))
    from common.hailo_logger import get_logger

logger = get_logger(__name__)

class UFLDProcessing:
    def __init__(self,
                 num_cell_row,
                 num_cell_col,
                 num_row,
                 num_col,
                 num_lanes,
                 crop_ratio,
                 original_frame_width,
                 original_frame_height,
                 total_frames):
        """
        Initialize UFLDProcessing with parameters.

        Args:
            num_cell_row (int): Number of rows for the grid cells.
            num_cell_col (int): Number of columns for the grid cells.
            num_row (int): Number of rows for lanes.
            num_col (int): Number of columns for lanes.
            num_lanes (int): Number of lanes.
            crop_ratio (float): Ratio for cropping the frame.
            original_frame_width (int): Width of the original image.
            original_frame_height (int): Height of the original image.
            total_frames (int): Total number of frames in the video.
        """
        self.num_cell_row = num_cell_row
        self.num_cell_col = num_cell_col
        self.num_row = num_row
        self.num_col = num_col
        self.num_lanes = num_lanes
        self.crop_ratio = crop_ratio
        self.original_frame_width = original_frame_width
        self.original_frame_height = original_frame_height
        self.total_frames = total_frames

    def resize(self, image, input_height, input_width):
        """
        Resize and crop an image.

        Args:
            image (numpy.ndarray): Input image.
            input_height (int): Height of the input frame.
            input_width (int): Width of the input frame.

        Returns:
            numpy.ndarray: Resized and cropped image.
        """
        new_height = int(input_height / self.crop_ratio)
        image_resized = cv2.resize(image, (input_width, new_height), interpolation=cv2.INTER_CUBIC)
        image_resized = image_resized[-320:, :, :]
        return image_resized

    def _soft_max(self, z):
        """
        Compute the softmax function for a given array.

        Args:
            z (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Softmax of the input array.
        """
        t = np.exp(z)
        a = np.exp(z) / np.sum(t)
        return a

    def _slice_and_reshape(self, output):
        """
        Slice and reshape the output tensor.

        Args:
            output (numpy.ndarray): Output tensor from inference.

        Returns:
            numpy.ndarray: Sliced and reshaped tensors for row and column information.
        """
        
        # Calculate the dimensions for slicing the output tensor
        dim1 = self.num_cell_row * self.num_row * self.num_lanes
        dim2 = self.num_cell_col * self.num_col * self.num_lanes
        dim3 = 2 * self.num_row * self.num_lanes
        dim4 = 2 * self.num_col * self.num_lanes

        # Slice and reshape the output tensor
        loc_row = np.reshape(output[:, :dim1],
                             (-1, self.num_cell_row, self.num_row, self.num_lanes))
        loc_col = np.reshape(output[:, dim1:dim1 + dim2],
                             (-1, self.num_cell_col, self.num_col, self.num_lanes))
        exist_row = np.reshape(output[:, dim1 + dim2:dim1 + dim2 + dim3],
                               (-1, 2, self.num_row, self.num_lanes))
        exist_col = np.reshape(output[:, -dim4:],
                               (-1, 2, self.num_col, self.num_lanes))
        return loc_row, loc_col, exist_row, exist_col

    def _pred2coords(self, loc_row, loc_col, exist_row, exist_col, local_width=1):
        """
        Convert prediction data to lane coordinates.

        Args:
            loc_row (numpy.ndarray): Row localization information.
            loc_col (numpy.ndarray): Column localization information.
            exist_row (numpy.ndarray): Existence of rows.
            exist_col (numpy.ndarray): Existence of columns.
            local_width (int): Local width for localization.

        Returns:
            list: List of lane coordinates.
        """

        row_anchor = np.linspace(160, 710, 56) / 720
        col_anchor = np.linspace(0, 1, 41)
        _, num_grid_row, num_cls_row, _ = loc_row.shape
        _, num_grid_col, num_cls_col, _ = loc_col.shape
        max_indices_row = np.argmax(loc_row, 1)
        valid_row = np.argmax(exist_row, 1)
        max_indices_col = np.argmax(loc_col, 1)
        valid_col = np.argmax(exist_col, 1)
        coords = []
        row_lane_idx = [1, 2]
        col_lane_idx = [0, 3]
        for i in row_lane_idx:
            tmp = []
            valid_row_sum = np.sum(valid_row[0, :, i])
            if valid_row_sum > num_cls_row / 2:
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:
                        all_ind_min = max(0, max_indices_row[0, k, i] - local_width)
                        all_ind_max = min(num_grid_row - 1,
                                        max_indices_row[0, k, i] + local_width) + 1
                        all_ind = list(range(all_ind_min, all_ind_max))
                        row_softmax = self._soft_max(loc_row[0, all_ind_min:all_ind_max, k, i])
                        out_tmp = np.sum(row_softmax * all_ind) + 0.5
                        out_tmp = out_tmp / (num_grid_row - 1) * self.original_frame_width
                        tmp.append((int(out_tmp), int(row_anchor[k] * self.original_frame_height)))
                coords.append(tmp)
        for i in col_lane_idx:
            tmp = []
            valid_col_sum = np.sum(valid_col[0, :, i])
            if valid_col_sum > (num_cls_col / 4):
                for k in range(valid_col.shape[1]):
                    if valid_col[0, k, i]:
                        all_ind_min = max(0, max_indices_col[0, k, i] - local_width)
                        all_ind_max = min(num_grid_col - 1,
                                        max_indices_col[0, k, i] + local_width) + 1
                        all_ind = list(range(all_ind_min, all_ind_max))
                        all_ind = range(all_ind_min, all_ind_max)
                        col_softmax = self._soft_max(loc_col[0, all_ind_min:all_ind_max, k, i])
                        out_tmp = np.sum(col_softmax * all_ind) + 0.5
                        out_tmp = out_tmp / (num_grid_col - 1) * self.original_frame_height
                        tmp.append((int(col_anchor[k] * self.original_frame_width), int(out_tmp)))
                coords.append(tmp)
        return coords

    def get_coordinates(self, endnodes):
        """
        Get lane coordinates from inference results.

        Args:
            endnodes (numpy.ndarray): Inference output.

        Returns:
            list: List of lane coordinates.
        """
        loc_row, loc_col, exist_row, exist_col = self._slice_and_reshape(endnodes)
        return self._pred2coords(loc_row, loc_col, exist_row, exist_col)


    def get_original_frame_size(self):
        """
        Retrieve the original frame size.

        Returns:
            tuple: A tuple containing the original frame width and height as (width, height).
        """
        return (self.original_frame_width, self.original_frame_height)



def check_process_errors(*processes: Process) -> None:
    """
    Check the exit codes of processes and log errors if any process has a non-zero exit code.

    Args:
        processes (Process): The processes to check.
    """
    process_failed = False
    for process in processes:
        if process.exitcode != 0:
            logger.error(f"{process.name} terminated with an error. Exit code: {process.exitcode}")
            process_failed = True
    if process_failed:
        raise RuntimeError("One or more processes terminated with an error.")


def configure_output_video(output_path,
                           frame_rate,
                           resolution) -> cv2.VideoWriter:
    """
    Configures the output video codec and creates an output video file.

    Args:
        output_path (str): Path to the output video file.
        frame_rate (int, optional): Frame rate for the video.
        resolution (tuple[int, int], optional): Resolution of the output video. 
    Returns:
        cv2.VideoWriter: VideoWriter object for writing frames to the output video.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, frame_rate, resolution)
    return output_video


def compute_scaled_radius(
    width: int,
    height: int,
    standard_width: int = 1280,
    standard_height: int = 720,
    base_radius: int = 5
) -> int:
    """
    Compute a scaled circle radius based on the video resolution.

    Args:
        width (int): Width of the input video frame.
        height (int): Height of the input video frame.
        standard_width (int, optional): Standard width for scaling reference. Defaults to 1280.
        standard_height (int, optional): Standard height for scaling reference. Defaults to 720.
        base_radius (int, optional): Base radius at standard resolution. Defaults to 5.

    Returns:
        int: Scaled radius for drawing circles.
    """
    standard_diag = hypot(standard_width, standard_height)
    diag = hypot(width, height)
    scale = diag / standard_diag
    radius = max(int(base_radius * scale), 1)  # Ensure radius is at least 1
    return radius
