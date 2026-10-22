import json
import logging
import os
import time
import uuid

import numpy as np
import torch
from PIL import Image

import folder_paths

from .unfake_core import unfake

logger = logging.getLogger(__name__)

# Largest side of the upscaled image / preview, to avoid multi-GB tensors
MAX_UPSCALED_SIDE = 4096


class UnfakePixelArtNode:
    """
    ComfyUI Node to convert fake AI pixel art to true pixel art.
    Based on the unfake.js algorithm: Edge-Aware Scale Detection -> Optimal Crop -> Downscale -> Quantization.
    """

    DESCRIPTION = (
        "Converts blurry / upscaled AI 'pixel art' into real pixel art: one pixel per art pixel, "
        "aligned to the grid, with a limited palette and optional transparent background."
    )

    # NOTE: saved workflows store widget values by position. Only append new
    # inputs at the end of this dict, never reorder or remove existing ones.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Exact output width in art pixels (e.g. 64). "
                        "0 = derive it from target_height keeping the aspect ratio. "
                        "Both 0 = auto-detect the pixel size.",
                    },
                ),
                "target_height": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Exact output height in art pixels (e.g. 64). "
                        "0 = derive it from target_width keeping the aspect ratio. "
                        "Both 0 = auto-detect the pixel size.",
                    },
                ),
                "manual_scale_factor": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 128,
                        "step": 1,
                        "tooltip": "Size of one art pixel in input pixels (e.g. 16 turns 1024 into 64). "
                        "0 = auto-detect. Only used when both target sizes are 0.",
                    },
                ),
                "remove_background": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Make the background transparent. The background color is taken "
                        "from the image border, and only regions connected to the border are removed.",
                    },
                ),
                "background_tolerance": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "tooltip": "Maximum per-channel difference from the background color. "
                        "Raise it for noisy / JPEG backgrounds.",
                    },
                ),
                "max_colors": (
                    "INT",
                    {
                        "default": 16,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "display": "number",
                        "tooltip": "Palette size. 0 = keep all colors.",
                    },
                ),
                "cleanup_jaggies": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Replace isolated stray pixels with the color of their neighbors "
                        "(never adds new colors). May remove intentional 1-pixel details.",
                    },
                ),
                "downscale_method": (
                    ["nearest", "dominant"],
                    {
                        "default": "nearest",
                        "tooltip": "nearest: center pixel of each cell (crisp, fast). "
                        "dominant: most common color of each cell (best for noisy AI images).",
                    },
                ),
                "scale_detection_method": (["edge_aware"], {"default": "edge_aware"}),
                "ea_tile_grid_size": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "display": "slider",
                        "tooltip": "Auto-detection: the image is split into NxN tiles and the most detailed half is analyzed.",
                    },
                ),
                "ea_min_peak_distance": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 50,
                        "step": 1,
                        "display": "number",
                        "tooltip": "Auto-detection: minimum distance between edges. Lower it if the art pixels are smaller than this.",
                    },
                ),
                "ea_peak_prominence_factor": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "number",
                        "tooltip": "Auto-detection: minimum edge strength, relative to the strongest edge.",
                    },
                ),
                "upscale_preview_factor": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                        "display": "number",
                        "tooltip": "Nearest-neighbor upscale for the node preview and the upscaled_image output "
                        f"(capped so the largest side stays under {MAX_UPSCALED_SIDE}px).",
                    },
                ),
                "fit_mode": (
                    ["crop", "pad", "stretch"],
                    {
                        "default": "crop",
                        "tooltip": "When the input aspect ratio differs from the target size. "
                        "crop: fill the target and cut the excess. "
                        "pad: fit inside and add transparent borders. "
                        "stretch: distort to fill.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "IMAGE")
    RETURN_NAMES = ("pixel_art_image", "manifest", "mask", "upscaled_image")
    OUTPUT_TOOLTIPS = (
        "Real-size pixel art. RGBA when the background is removed or padding is added, RGB otherwise.",
        "JSON with the detected pixel size, grid offset, output size and palette.",
        "ComfyUI-style mask of the transparent area (1 = transparent), as returned by Load Image.",
        "The pixel art upscaled with nearest neighbor, ready to view or save.",
    )
    FUNCTION = "process"
    CATEGORY = "image/postprocessing"

    def process(
        self,
        image: torch.Tensor,
        target_width: int,
        target_height: int,
        manual_scale_factor: int,
        remove_background: bool,
        background_tolerance: int,
        max_colors: int,
        cleanup_jaggies: bool,
        downscale_method: str,
        scale_detection_method: str,
        ea_tile_grid_size: int,
        ea_min_peak_distance: int,
        ea_peak_prominence_factor: float,
        upscale_preview_factor: int,
        fit_mode: str = "crop",
    ):
        start_time = time.time()
        frames = (image.detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)

        has_target = target_width > 0 or target_height > 0
        scale = 0 if has_target else manual_scale_factor
        results, infos = [], []
        for i, frame in enumerate(frames):
            result, info = unfake(
                frame,
                target_width=target_width,
                target_height=target_height,
                fit_mode=fit_mode,
                scale=scale,
                remove_bg=remove_background,
                background_tolerance=background_tolerance,
                max_colors=max_colors,
                cleanup=cleanup_jaggies,
                downscale_method=downscale_method,
                tile_grid_size=ea_tile_grid_size,
                min_peak_distance=ea_min_peak_distance,
                peak_prominence_factor=ea_peak_prominence_factor,
            )
            # Reuse the first frame's detected scale so a batch (e.g. animation frames) stays consistent
            if i == 0 and not has_target and scale <= 0:
                scale = int(info["pixel_size"][0])
            results.append(result)
            infos.append(info)

        # Grid offsets can differ by one pixel between frames; crop to a common size
        out_h = min(r.shape[0] for r in results)
        out_w = min(r.shape[1] for r in results)
        batch = np.stack([r[:out_h, :out_w] for r in results])

        use_alpha = (
            remove_background
            or (has_target and fit_mode == "pad")
            or image.shape[-1] == 4
            or bool((batch[..., 3] < 255).any())
        )
        if not use_alpha:
            batch = batch[..., :3]

        output = torch.from_numpy(batch.astype(np.float32) / 255.0)
        mask = 1.0 - torch.from_numpy(batch[..., 3].astype(np.float32) / 255.0) if use_alpha else torch.zeros(batch.shape[:3])

        factor = max(1, min(upscale_preview_factor, MAX_UPSCALED_SIDE // max(out_w, out_h)))
        upscaled = output.repeat_interleave(factor, dim=1).repeat_interleave(factor, dim=2)

        manifest = json.dumps(
            {
                "frames": len(infos),
                "processing_time_ms": round((time.time() - start_time) * 1000),
                "preview_factor": factor,
                **(infos[0] if len(infos) == 1 else {"per_frame": infos}),
            },
            indent=2,
        )
        logger.info("[Unfake Pixel Art] %s", json.dumps({k: v for k, v in infos[0].items() if k != "palette"}))

        return {"ui": {"images": self.save_previews(batch, factor)}, "result": (output, manifest, mask, upscaled)}

    def save_previews(self, batch: np.ndarray, factor: int) -> list:
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        run_id = uuid.uuid4().hex[:10]
        previews = []
        for i, frame in enumerate(batch):
            img = Image.fromarray(frame)
            img = img.resize((img.width * factor, img.height * factor), Image.Resampling.NEAREST)
            filename = f"unfake_preview_{run_id}_{i:03d}.png"
            img.save(os.path.join(temp_dir, filename), compress_level=1)
            previews.append({"filename": filename, "subfolder": "", "type": "temp"})
        return previews


NODE_CLASS_MAPPINGS = {
    "UnfakePixelArtNode": UnfakePixelArtNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnfakePixelArtNode": "Unfake Pixel Art (AI to Real)",
}
