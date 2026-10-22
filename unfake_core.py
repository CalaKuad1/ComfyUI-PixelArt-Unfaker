"""
Core "unfake" algorithm, independent of ComfyUI.

All functions work on RGBA uint8 numpy arrays of shape (H, W, 4).
Pipeline: background removal -> scale detection -> grid alignment ->
grid downscale -> fit to target -> color quantization -> cleanup.
"""

import logging
import math
from collections import Counter

import numpy as np
from PIL import Image
from scipy.ndimage import label, sobel

logger = logging.getLogger(__name__)

ALPHA_THRESHOLD = 128
TRANSPARENT_KEY = -1  # packed color key used for transparent pixels
OUTSIDE_KEY = -2  # packed color key used for out-of-image neighbors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_rgba(arr: np.ndarray) -> np.ndarray:
    """Converts an (H, W, C) uint8 array with 1, 3 or 4 channels to RGBA."""
    if arr.ndim == 2:
        arr = arr[..., None]
    c = arr.shape[2]
    if c == 4:
        return arr.copy()
    if c == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif c != 3:
        raise ValueError(f"Unsupported number of channels ({c})")
    alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.concatenate([arr, alpha], axis=2)


def gray_on_white(arr: np.ndarray) -> np.ndarray:
    """Grayscale float image with transparency composited over white."""
    rgb = arr[..., :3].astype(np.float32)
    a = arr[..., 3:4].astype(np.float32) / 255.0
    rgb = rgb * a + 255.0 * (1.0 - a)
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def edge_profiles(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise and row-wise sums of absolute Sobel gradients."""
    profile_x = np.abs(sobel(gray, axis=1, mode="nearest")).sum(axis=0)
    profile_y = np.abs(sobel(gray, axis=0, mode="nearest")).sum(axis=1)
    return profile_x, profile_y


def pack_keys(arr: np.ndarray) -> np.ndarray:
    """Packs RGB into one int64 per pixel; transparent pixels get TRANSPARENT_KEY."""
    rgb = arr[..., :3].astype(np.int64)
    keys = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    return np.where(arr[..., 3] >= ALPHA_THRESHOLD, keys, TRANSPARENT_KEY)


def unpack_keys(keys: np.ndarray) -> np.ndarray:
    """Inverse of pack_keys."""
    out = np.zeros(keys.shape + (4,), dtype=np.uint8)
    opaque = keys >= 0
    k = np.where(opaque, keys, 0)
    out[..., 0] = (k >> 16) & 255
    out[..., 1] = (k >> 8) & 255
    out[..., 2] = k & 255
    out[..., 3] = np.where(opaque, 255, 0)
    return out


def row_mode(keys: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Most frequent value along the last axis, considering only `valid` entries.
    Returns (mode, count); rows without valid entries get (OUTSIDE_KEY, 0).
    """
    s = np.sort(np.where(valid, keys, OUTSIDE_KEY), axis=-1)
    n = s.shape[-1]
    pos = np.arange(n)
    new_run = np.ones(s.shape, dtype=bool)
    new_run[..., 1:] = s[..., 1:] != s[..., :-1]
    run_start = np.maximum.accumulate(np.where(new_run, pos, 0), axis=-1)
    run_len = np.where(s == OUTSIDE_KEY, 0, pos - run_start + 1)
    best = run_len.argmax(axis=-1)[..., None]
    mode = np.take_along_axis(s, best, axis=-1)[..., 0]
    count = np.take_along_axis(run_len, best, axis=-1)[..., 0]
    return mode, count


# ---------------------------------------------------------------------------
# Background removal
# ---------------------------------------------------------------------------


def remove_background(arr: np.ndarray, tolerance: int) -> np.ndarray:
    """
    Makes transparent the background region connected to the image border.
    The background color is the median of the opaque border pixels, so it
    works with noisy / JPEG-compressed backgrounds.
    """
    border = np.concatenate([arr[0], arr[-1], arr[1:-1, 0], arr[1:-1, -1]])
    opaque_border = border[border[:, 3] >= ALPHA_THRESHOLD]
    if len(opaque_border) == 0:
        return arr.copy()  # already transparent

    bg_color = np.median(opaque_border[:, :3], axis=0).astype(np.int16)
    diff = np.abs(arr[..., :3].astype(np.int16) - bg_color)
    mask = np.all(diff <= tolerance, axis=-1) & (arr[..., 3] >= ALPHA_THRESHOLD)

    # Only erase regions touching the border, not similar colors inside the sprite
    labels, _ = label(mask, structure=np.ones((3, 3), dtype=np.int32))
    edge_labels = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    edge_labels = edge_labels[edge_labels != 0]

    out = arr.copy()
    if len(edge_labels):
        out[np.isin(labels, edge_labels), 3] = 0
    return out


# ---------------------------------------------------------------------------
# Scale detection
# ---------------------------------------------------------------------------


def _find_peaks(profile: np.ndarray, min_dist: int, prom_factor: float) -> list:
    if len(profile) < 3:
        return []
    max_val = profile.max()
    if max_val <= 0:
        return []
    p = profile
    is_peak = (p[1:-1] > p[:-2]) & (p[1:-1] > p[2:]) & (p[1:-1] >= max_val * prom_factor)
    peaks = (np.nonzero(is_peak)[0] + 1).tolist()

    filtered = peaks[:1]
    for peak in peaks[1:]:
        if peak - filtered[-1] >= min_dist:
            filtered.append(peak)
    return filtered


def detect_scale(
    arr: np.ndarray,
    tile_grid_size: int = 3,
    min_peak_distance: int = 5,
    peak_prominence_factor: float = 0.1,
) -> int:
    """
    Edge-aware scale detection (unfake.js): measures the spacing between
    gradient peaks on the most detailed tiles and takes the GCD / mode.
    Returns 1 when no reliable scale is found.
    """
    gray = gray_on_white(arr)
    h, w = gray.shape
    tile_h, tile_w = h // tile_grid_size, w // tile_grid_size

    if tile_h <= 1 or tile_w <= 1:
        tiles = [gray]
    else:
        tiles = [
            gray[i * tile_h : (i + 1) * tile_h, j * tile_w : (j + 1) * tile_w]
            for i in range(tile_grid_size)
            for j in range(tile_grid_size)
        ]
        tiles.sort(key=np.var, reverse=True)
        tiles = tiles[: max(1, len(tiles) // 2)]

    distances = []
    for tile in tiles:
        if tile.shape[0] < 10 or tile.shape[1] < 10:
            continue
        for profile in edge_profiles(tile):
            peaks = _find_peaks(profile, min_peak_distance, peak_prominence_factor)
            distances.extend(np.diff(peaks).tolist())

    distances = [d for d in distances if d >= 2]
    if not distances:
        return 1

    counts = Counter(distances)
    common = [d for d, _ in counts.most_common(20)]
    scale_mode = common[0]

    gcd_val = common[0]
    for d in common[1:]:
        gcd_val = math.gcd(gcd_val, d)
        if gcd_val == 1:
            break

    max_reasonable = max(2, min(w, h) // 4)
    if 2 <= gcd_val <= max_reasonable:
        return gcd_val
    if 2 <= scale_mode <= max_reasonable:
        return scale_mode
    return 1


# ---------------------------------------------------------------------------
# Grid alignment and downscaling
# ---------------------------------------------------------------------------


def best_grid_offset(profile: np.ndarray, cell: float, lo: int, hi: int) -> int:
    """
    Integer offset in [lo, hi] whose grid lines (offset + k * cell) land on
    the most edge energy. Uses the mean so offsets are not penalized for
    covering fewer lines.
    """
    n = len(profile)
    lo, hi = max(0, lo), min(hi, n - 1)
    if cell <= 1 or hi <= lo:
        return max(0, lo)

    best_offset, best_score = lo, -1.0
    for offset in range(lo, hi + 1):
        idx = np.round(offset + np.arange(0, (n - offset) / cell) * cell).astype(int)
        idx = idx[idx < n]
        if len(idx) == 0:
            continue
        score = float(profile[idx].mean())
        if score > best_score:
            best_offset, best_score = offset, score
    return best_offset


def _dominant_color(blocks: np.ndarray) -> np.ndarray:
    """
    blocks: (..., N, 4) -> (..., 4).
    Majority vote on alpha; among opaque pixels, picks the most common
    coarse color bucket (5 bits per channel) and averages its members.
    Robust to AI noise, where almost every pixel has a unique exact color.
    """
    opaque = blocks[..., 3] >= ALPHA_THRESHOLD
    n_opaque = opaque.sum(axis=-1)

    rgb = blocks[..., :3].astype(np.int64)
    coarse = rgb >> 3
    keys = (coarse[..., 0] << 10) | (coarse[..., 1] << 5) | coarse[..., 2]
    mode, _ = row_mode(keys, opaque)

    members = (keys == mode[..., None]) & opaque
    weight = np.maximum(members.sum(axis=-1), 1)[..., None]
    mean = (rgb * members[..., None]).sum(axis=-2) / weight

    out = np.zeros(blocks.shape[:-2] + (4,), dtype=np.uint8)
    is_opaque = n_opaque * 2 > blocks.shape[-2]
    out[..., :3] = np.where(is_opaque[..., None], np.round(mean), 0).astype(np.uint8)
    out[..., 3] = np.where(is_opaque, 255, 0)
    return out


def sample_grid(
    arr: np.ndarray,
    x0: float,
    y0: float,
    cell_w: float,
    cell_h: float,
    out_w: int,
    out_h: int,
    method: str = "nearest",
) -> np.ndarray:
    """
    Downscales by reading one output pixel per grid cell. Cells may have a
    fractional size, so any target resolution can be reached without
    stretching or half pixels.
    """
    h, w = arr.shape[:2]
    starts_x = x0 + np.arange(out_w) * cell_w
    starts_y = y0 + np.arange(out_h) * cell_h

    if method == "nearest":
        xi = np.clip(np.floor(starts_x + cell_w / 2).astype(int), 0, w - 1)
        yi = np.clip(np.floor(starts_y + cell_h / 2).astype(int), 0, h - 1)
        return arr[yi[:, None], xi[None, :]]

    # "dominant": vote over the inner 70% of each cell (cell borders are blurry)
    kx, ky = max(1, round(cell_w * 0.7)), max(1, round(cell_h * 0.7))
    inset_x, inset_y = (cell_w - kx) / 2, (cell_h - ky) / 2
    xb = np.clip(np.floor(starts_x + inset_x).astype(int)[:, None] + np.arange(kx), 0, w - 1)
    yb = np.clip(np.floor(starts_y + inset_y).astype(int)[:, None] + np.arange(ky), 0, h - 1)

    out = np.empty((out_h, out_w, 4), dtype=np.uint8)
    rows_per_chunk = max(1, 2_000_000 // (out_w * kx * ky))
    for r0 in range(0, out_h, rows_per_chunk):
        yc = yb[r0 : r0 + rows_per_chunk]
        blocks = arr[yc[:, None, :, None], xb[None, :, None, :]]  # (r, out_w, ky, kx, 4)
        blocks = blocks.reshape(len(yc), out_w, ky * kx, 4)
        out[r0 : r0 + len(yc)] = _dominant_color(blocks)
    return out


# ---------------------------------------------------------------------------
# Color quantization
# ---------------------------------------------------------------------------


def _sq_distances(a, b):
    """Squared Euclidean distances between rows of a (M, 3) and b (K, 3)."""
    return (a * a).sum(1)[:, None] - 2.0 * a @ b.T + (b * b).sum(1)[None, :]


def _kmeans_refine(colors, weights, centers, iterations=10):
    """Weighted Lloyd iterations on unique colors."""
    colors = colors.astype(np.float64)
    for _ in range(iterations):
        assign = _sq_distances(colors, centers).argmin(axis=1)
        new_centers = centers.copy()
        for k in range(len(centers)):
            sel = assign == k
            if sel.any():
                wk = weights[sel]
                new_centers[k] = (colors[sel] * wk[:, None]).sum(axis=0) / wk.sum()
        if np.allclose(new_centers, centers, atol=0.5):
            centers = new_centers
            break
        centers = new_centers
    return centers


def _nearest_center(colors, centers, chunk=16384):
    out = np.empty(len(colors), dtype=np.int64)
    c = centers.astype(np.float64)
    for i in range(0, len(colors), chunk):
        out[i : i + chunk] = _sq_distances(colors[i : i + chunk].astype(np.float64), c).argmin(1)
    return out


def quantize(arr: np.ndarray, max_colors: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduces opaque pixels to at most `max_colors` colors (median cut + k-means)
    and binarizes alpha. Transparent pixels never consume palette entries.
    Returns (image, palette as (K, 3) uint8).
    """
    out = arr.copy()
    opaque = out[..., 3] >= ALPHA_THRESHOLD
    out[..., 3] = np.where(opaque, 255, 0)
    out[~opaque, :3] = 0

    pixels = out[opaque][:, :3]
    if len(pixels) == 0:
        return out, np.zeros((0, 3), dtype=np.uint8)

    uniq, inverse, counts = np.unique(pixels, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    if max_colors <= 0 or len(uniq) <= max_colors:
        return out, uniq.astype(np.uint8)

    # Initial palette: Pillow median cut over the opaque pixels only
    strip = Image.fromarray(np.ascontiguousarray(pixels.reshape(1, -1, 3)))
    q = strip.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    n_used = int(np.asarray(q).max()) + 1
    centers = np.array(q.getpalette()[: 3 * n_used], dtype=np.float64).reshape(-1, 3)

    # Refine with k-means on (a sample of) the unique colors
    fit_idx = np.arange(len(uniq))
    if len(fit_idx) > 20000:
        fit_idx = np.random.default_rng(0).choice(len(uniq), 20000, replace=False)
    centers = _kmeans_refine(uniq[fit_idx], counts[fit_idx].astype(np.float64), centers)

    palette = np.clip(np.round(centers), 0, 255).astype(np.uint8)
    mapped = palette[_nearest_center(uniq, palette)]
    out[opaque, :3] = mapped[inverse]

    used = np.unique(out[opaque][:, :3], axis=0)
    return out, used


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_isolated_pixels(arr: np.ndarray, min_agreement: int = 6) -> np.ndarray:
    """
    Replaces pixels that match none of their 8 neighbors when at least
    `min_agreement` of those neighbors share one color. The replacement is
    always an existing color (or transparency), so the palette is preserved.
    Also removes lone opaque specks left by background removal.
    """
    keys = pack_keys(arr)
    h, w = keys.shape
    padded = np.pad(keys, 1, constant_values=OUTSIDE_KEY)
    shifts = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    neighbors = np.stack(
        [padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy, dx in shifts], axis=-1
    )

    matches_self = (neighbors == keys[..., None]).any(axis=-1)
    mode, count = row_mode(neighbors, neighbors != OUTSIDE_KEY)
    replace = ~matches_self & (count >= min_agreement)

    out = arr.copy()
    out[replace] = unpack_keys(mode[replace])
    return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def resolve_target(w: int, h: int, target_w: int, target_h: int) -> tuple[int, int]:
    """Fills a 0 target dimension from the other one, keeping aspect ratio."""
    if target_w > 0 and target_h <= 0:
        target_h = max(1, round(h * target_w / w))
    elif target_h > 0 and target_w <= 0:
        target_w = max(1, round(w * target_h / h))
    return target_w, target_h


def unfake(
    arr: np.ndarray,
    target_width: int = 0,
    target_height: int = 0,
    fit_mode: str = "crop",
    scale: int = 0,
    remove_bg: bool = True,
    background_tolerance: int = 10,
    max_colors: int = 16,
    cleanup: bool = True,
    downscale_method: str = "nearest",
    tile_grid_size: int = 3,
    min_peak_distance: int = 5,
    peak_prominence_factor: float = 0.1,
) -> tuple[np.ndarray, dict]:
    """
    Runs the whole pipeline on one image. `arr` is (H, W, C) uint8.
    `scale` > 0 forces the pixel size; 0 auto-detects it. Target dimensions
    (when set) take precedence over `scale`.
    Returns the RGBA result and a dict describing what was done.
    """
    arr = to_rgba(arr)
    if remove_bg:
        arr = remove_background(arr, background_tolerance)

    h, w = arr.shape[:2]
    profile_x, profile_y = edge_profiles(gray_on_white(arr))
    target_w, target_h = resolve_target(w, h, target_width, target_height)
    info = {"input_size": [w, h]}

    if target_w > 0 and target_h > 0:
        if fit_mode == "stretch":
            cell_w, cell_h = w / target_w, h / target_h
            out_w, out_h = target_w, target_h
        else:
            cell = min(w / target_w, h / target_h) if fit_mode == "crop" else max(w / target_w, h / target_h)
            cell_w = cell_h = cell
            out_w = min(target_w, int(w / cell + 1e-6))
            out_h = min(target_h, int(h / cell + 1e-6))
        # Align the grid to the image edges, staying close to a centered crop
        slack_x, slack_y = w - out_w * cell_w, h - out_h * cell_h
        cx, cy = slack_x / 2, slack_y / 2
        x0 = best_grid_offset(profile_x, cell_w, math.floor(cx - cell_w / 2), math.ceil(cx + cell_w / 2))
        y0 = best_grid_offset(profile_y, cell_h, math.floor(cy - cell_h / 2), math.ceil(cy + cell_h / 2))
        x0, y0 = min(x0, max(0, math.floor(slack_x))), min(y0, max(0, math.floor(slack_y)))
        info["mode"] = f"target ({fit_mode})"
    else:
        if scale <= 0:
            scale = detect_scale(arr, tile_grid_size, min_peak_distance, peak_prominence_factor)
            info["mode"] = "auto-detected scale"
        else:
            info["mode"] = "manual scale"
        cell_w = cell_h = float(scale)
        x0 = best_grid_offset(profile_x, scale, 0, scale - 1)
        y0 = best_grid_offset(profile_y, scale, 0, scale - 1)
        out_w, out_h = max(1, (w - x0) // scale), max(1, (h - y0) // scale)

    result = sample_grid(arr, x0, y0, cell_w, cell_h, out_w, out_h, downscale_method)

    # Pad mode: center the content on a transparent canvas of the exact target size
    if target_w > 0 and target_h > 0 and (out_w, out_h) != (target_w, target_h):
        canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        ox, oy = (target_w - out_w) // 2, (target_h - out_h) // 2
        canvas[oy : oy + out_h, ox : ox + out_w] = result
        result = canvas

    result, palette = quantize(result, max_colors)

    if cleanup and (cell_w > 1 or cell_h > 1):
        result = cleanup_isolated_pixels(result)
        palette = np.unique(result[result[..., 3] > 0][:, :3], axis=0)

    info.update(
        {
            "pixel_size": [round(cell_w, 3), round(cell_h, 3)],
            "grid_offset": [int(x0), int(y0)],
            "output_size": [int(result.shape[1]), int(result.shape[0])],
            "colors": int(len(palette)),
            "palette": ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in palette],
        }
    )
    return result, info
