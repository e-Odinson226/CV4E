import numpy as np
from typing import Optional

def compute_patch_budget(
    total_context: int,
    gaze_ratio: float,
) -> tuple[int, int]:
    """
    Calculating how many patches from uniform sampling and how many from sampling guided by gaze

    Invariant: n_uniform + n_gaze == total_context.

    Args:
        total_context: total number of context patches
        gaze_ratio:    how much allocated for gaze sampling

    Returns:
        (n_uniform, n_gaze): numbers of patches from each

    Raises:
        ValueError: if gaze ratio not between 0 and 1
        ValueError: if total context <0
    """
    if not (0.0 <= gaze_ratio <= 1.0):
        raise ValueError(
            f"gaze_ratio must me between [0,1]: {gaze_ratio}"
        )
    if total_context <= 0:
        raise ValueError(
            f"total context must be positive, not: {total_context}"
        )

    n_gaze    = round(total_context * gaze_ratio)
    n_uniform = total_context - n_gaze 
    return n_uniform, n_gaze


def sample_uniform_patches(
    total_patches: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Selecting patches with uniform distribution. Standard V-Jepa

    Args:
        total_patches: patches available for the timestamp
        n_samples:     how many patches to select
        rng:           numpy random Generator

    Returns:
        indices: np.ndarray shape (n_samples,), dtype int64

    Raises:
        ValueError: if n_samples > total_patches
    """
    if n_samples > total_patches:
        raise ValueError(
            f"Can select this many patches {n_samples} from {total_patches} patches available."
        )
    if n_samples == 0:
        return np.array([], dtype=np.int64)

    return rng.choice(total_patches, size=n_samples, replace=False)



def sample_gaze_patches(
    prob_map: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Selects gaze influenced patches or uniform fallback
    Sampling without replacement to not have duplicates
    Args:
        prob_map:  np.ndarray shape (grid_h, grid_w), sum=1.0
                   Probability mask
        n_samples: how many patches
        rng:       numpy random Generator

    Returns:
        indices: np.ndarray shape (n_samples,), dtype int64

    Raises:
        ValueError: if n_samples > total patches from prob_map
    """
    total_patches = prob_map.size  
    if n_samples > total_patches:
        raise ValueError(
            f"Can not select {n_samples} patches from {total_patches}."
        )
    if n_samples == 0:
        return np.array([], dtype=np.int64)

    flat_probs = prob_map.flatten().astype(np.float64)
    #Normalise again to avoid error
    flat_probs /= flat_probs.sum()

    return rng.choice(total_patches, size=n_samples, replace=False, p=flat_probs)



def merge_patch_sets(
    uniform_indices: np.ndarray,
    gaze_indices: np.ndarray,
    total_patches: int,
    total_context: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Merging the sets - when overlayed complete with uniform distribution

    Args:
        uniform_indices: uniform selected indices
        gaze_indices:    -//- with gaze mapping
        total_patches:   total patches
        total_context:   context patches
        rng:             numpy random Generator

    Returns:
        context_indices: np.ndarray shape (total_context,), dtype int64
    """
    #to eliminate duplicates
    combined = np.union1d(uniform_indices, gaze_indices)

    if len(combined) == total_context:
        return combined

    if len(combined) > total_context:
        #rare case of having more patches then needed
        return rng.choice(combined, size=total_context, replace=False)

    n_missing = total_context - len(combined)
    all_indices = np.arange(total_patches, dtype=np.int64)
    unused = np.setdiff1d(all_indices, combined)

    if len(unused) < n_missing:
        #Not enough patches
        raise RuntimeError(
        f"Insufficient patches to fill context. "
        f"Need {n_missing} more patches but only {len(unused)} available."
    )

    fill = rng.choice(unused, size=n_missing, replace=False)
    return np.concatenate([combined, fill])



def compute_target_indices(
    context_indices: np.ndarray,
    total_patches: int,
) -> np.ndarray:
    """
    Calculates target patches

    """
    all_indices = np.arange(total_patches, dtype=np.int64)
    return np.setdiff1d(all_indices, context_indices)


def select_context_and_target_for_timestep(
    prob_map: np.ndarray,
    grid_h: int,
    grid_w: int,
    total_context: int,
    gaze_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Selection of context/target for one timeframe

    Args:
        prob_map:      shape (grid_h, grid_w), sum=1.0
        grid_h, grid_w: dimension for matrix
        total_context: how many context patches
        gaze_ratio:    how many gaze chosen patches ratio
        rng:           numpy random Generator

    Returns:
        (context_indices, target_indices):
            context_indices: np.ndarray shape (total_context,)
            target_indices:  np.ndarray shape (total_patches - total_context,)
    """
    total_patches = grid_h * grid_w


    n_uniform, n_gaze = compute_patch_budget(total_context, gaze_ratio)

    uniform_idx = sample_uniform_patches(total_patches, n_uniform, rng)

    gaze_idx = sample_gaze_patches(prob_map, n_gaze, rng)

    context_idx = merge_patch_sets(
        uniform_idx, gaze_idx, total_patches, total_context, rng
    )

    target_idx = compute_target_indices(context_idx, total_patches)

    return context_idx, target_idx



def select_context_and_target_for_clip(
    prob_maps: list[np.ndarray],
    grid_h: int,
    grid_w: int,
    total_context_per_timestep: int,
    gaze_ratio: float,
    seed: Optional[int] = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Selects for complete clip

    Args:
        prob_maps:  list of probability maps
        grid_h, grid_w:              grid dimension
        total_context_per_timestep:  how many patches context per frame
        gaze_ratio:                  how many gaze induced
        seed:                        seed for reproductibility

    Returns:
        (all_context, all_target):
            all_context: total_context list of T arrays
            all_target:  total_patches - total_context list of T arrays
    """
    rng = np.random.default_rng(seed)

    all_context = []
    all_target  = []

    for t, prob_map in enumerate(prob_maps):
        context_idx, target_idx = select_context_and_target_for_timestep(
            prob_map=prob_map,
            grid_h=grid_h,
            grid_w=grid_w,
            total_context=total_context_per_timestep,
            gaze_ratio=gaze_ratio,
            rng=rng,
        )
        all_context.append(context_idx)
        all_target.append(target_idx)

    return all_context, all_target