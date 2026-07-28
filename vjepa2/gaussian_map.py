
import numpy as np
from typing import Optional

def project_gaze_to_native_pixel(
    yaw: float,
    pitch: float,
    fu: float,
    fv: float,
    cx_native: float,
    cy_native: float,
) -> tuple[float, float]:
    """
    Convertește un unghi de privire (yaw, pitch în radiani, sistem CPF)
    în coordonate pixel pe senzorul nativ (2560×1920).
    Converts a gaze angle - pitch and yaw - intro a system for the sensor 2650x1920

    Args:
        yaw:        horizontal angle
        pitch:      uvertical angle
        fu, fv:     focal distance
        cx_native:  center point for horizontal view
        cy_native:  center point for vertical view

    Returns:
        (px, py): coordinates pixels for server
    """
    px = cx_native + fu * np.tan(yaw)
    py = cy_native - fv * (np.tan(pitch) / np.cos(yaw))
    return px, py


def scale_pixel_to_video(
    px_native: float,
    py_native: float,
    cx_native: float,
    cy_native: float,
    video_w: int,
    video_h: int,
) -> tuple[float, float]:
    """
    Scaling the coordinates from the sensor to video space 1280x960

    Args:
        px_native, py_native: sensor coordinates
        cx_native, cy_native: center points
        video_w, video_h:     video resolution pixels

    Returns:
        (px_video, py_video): scaled coordinates
    """
    sx = video_w  / (2.0 * cx_native)
    sy = video_h / (2.0 * cy_native)
    return px_native * sx, py_native * sy

def normalize_pixel_to_unit(
    px_video: float,
    py_video: float,
    video_w: int,
    video_h: int,
) -> tuple[float, float]:
    """
    Convertește coordonate pixel video în coordonate normalizate [0, 1].
    Converts the video ixel coordinates into normalised coordinates [0,1].

    Results (0.0, 0.0) = upper left corner.
    Results (1.0, 1.0) = lower right corner.

    Args:
        px_video, py_video: coordinates pixels in video space
        video_w, video_h:   video resolution

    Returns:
        (gx, gy): normalised coordinates
    """
    gx = px_video / video_w
    gy = py_video / video_h
    return gx, gy

def unit_to_grid(
    gx: float,
    gy: float,
    grid_w: int,
    grid_h: int,
) -> tuple[float, float]:
    """
    Convert the coordinates to the patches matrix
    Args:
        gx, gy:          normalised fixation
        grid_w, grid_h:  dimension for patches

    Returns:
        (cx_grid, cy_grid): center of fixation in patched frame
    """
    cx_grid = gx * grid_w  
    cy_grid = gy * grid_h
    return cx_grid, cy_grid

def compute_elliptic_distance_map(
    cx_grid: float,
    cy_grid: float,
    grid_w: int,
    grid_h: int,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    """
    Calculate eliptic distance
    Args:
        cx_grid, cy_grid:    center of fixation
        grid_w, grid_h:      dimension patched frame
        sigma_x:             horizontal length elipse
        sigma_y:             vertical length elipse

    Returns:
        dist_map: np.ndarray de shape (grid_h, grid_w), dtype float32
    """
    cols = np.arange(grid_w, dtype=np.float32)   # [0, 1, ..., W-1]
    rows = np.arange(grid_h, dtype=np.float32)   # [0, 1, ..., H-1]

    J, I = np.meshgrid(cols, rows)

    dx = (J - cx_grid) / sigma_x   
    dy = (I - cy_grid) / sigma_y  

    dist_map = np.sqrt(dx**2 + dy**2)
    return dist_map


def laplacian_weights_from_distance(dist_map: np.ndarray) -> np.ndarray:
    """
    Args:
        dist_map: np.ndarray shape (H, W)

    Returns:
        weights: np.ndarray shape (H, W), float32
    """
    return np.exp(-np.sqrt(dist_map)).astype(np.float32)


def normalize_to_probability(weights: np.ndarray) -> np.ndarray:
    """
    Normalising the weights map to a distribution of probability
    Args:
        weights: np.ndarray shape (H, W)

    Returns:
        prob_map: np.ndarray shape (H, W), sum = 1.0

    Raises:
        ValueError:is sum 0 or negative
    """
    total = weights.sum()
    if total <= 0.0:
        raise ValueError(
            f"Suma greutăților este {total:.6f}. "
            "Greutățile trebuie să fie pozitive."
        )
    return (weights / total).astype(np.float32)


def uniform_probability_map(grid_w: int, grid_h: int) -> np.ndarray:
    """
    Fallback when gaze is 0 
    """
    n = grid_h * grid_w
    return np.full((grid_h, grid_w), fill_value=1.0 / n, dtype=np.float32)



def build_gaze_probability_map(
    yaw: Optional[float],
    pitch: Optional[float],
    gaze_valid: bool,
    fu: float,
    fv: float,
    cx_native: float,
    cy_native: float,
    video_w: int,
    video_h: int,
    grid_w: int,
    grid_h: int,
    sigma_x_patches: float = 3.0,
    sigma_y_patches: float = 2.0,
) -> np.ndarray:
    """
    Principal function combining the steps

    """
   
    if not gaze_valid or yaw is None or pitch is None:
        return uniform_probability_map(grid_w, grid_h)

    if np.isnan(yaw) or np.isnan(pitch):
        return uniform_probability_map(grid_w, grid_h)

    px_native, py_native = project_gaze_to_native_pixel(
        yaw, pitch, fu, fv, cx_native, cy_native
    )

    px_video, py_video = scale_pixel_to_video(
        px_native, py_native, cx_native, cy_native, video_w, video_h
    )

    if not (0.0 <= px_video < video_w and 0.0 <= py_video < video_h):
        return uniform_probability_map(grid_w, grid_h)

    gx, gy = normalize_pixel_to_unit(px_video, py_video, video_w, video_h)

    cx_grid, cy_grid = unit_to_grid(gx, gy, grid_w, grid_h)

    dist_map = compute_elliptic_distance_map(
        cx_grid, cy_grid, grid_w, grid_h, sigma_x_patches, sigma_y_patches
    )

    weights = laplacian_weights_from_distance(dist_map)

    prob_map = normalize_to_probability(weights)

    return prob_map








