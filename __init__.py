import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
import logging
import time
from collections import Counter
import math
from scipy.ndimage import sobel, label
import folder_paths
import os
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnfakePixelArtNode:
    """
    ComfyUI Node to convert fake AI pixel art to true pixel art.
    Based on the unfake.js algorithm: Edge-Aware Scale Detection -> Optimal Crop -> Downscale -> Quantization.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image from ComfyUI (tensor B,H,W,C)
                "target_width": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Ancho exacto final (ej. 64). 0 para mantener original.",
                    },
                ),
                "target_height": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Alto exacto final (ej. 64). 0 para mantener original.",
                    },
                ),
                "manual_scale_factor": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 128,
                        "step": 1,
                        "tooltip": "Fuerza la escala (ej. 16 para 1024->64). Ignorado si target_width > 0.",
                    },
                ),
                "remove_background": ("BOOLEAN", {"default": True}),
                "background_tolerance": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "tooltip": "Tolerancia de color para quitar el fondo.",
                    },
                ),
                "max_colors": (
                    "INT",
                    {
                        "default": 16,
                        "min": 2,
                        "max": 256,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "cleanup_jaggies": ("BOOLEAN", {"default": True}),
                "downscale_method": (["nearest", "dominant"], {"default": "nearest"}),
                "scale_detection_method": (["edge_aware"], {"default": "edge_aware"}),
                "ea_tile_grid_size": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "display": "slider",
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
                        "tooltip": "Escala la imagen de salida para la previsualización",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("pixel_art_image", "manifest")
    FUNCTION = "process"
    CATEGORY = "image/postprocessing"

    def tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        try:
            image_tensor = image_tensor.squeeze(0)
            image_tensor = torch.clamp(image_tensor, 0, 1)
            image_np = (image_tensor.numpy() * 255).astype(np.uint8)

            h, w, c = image_np.shape

            if c == 1:
                pil_image = Image.fromarray(image_np[:, :, 0], mode="L").convert("RGB")
            elif c == 3:
                pil_image = Image.fromarray(image_np, mode="RGB")
            elif c == 4:
                pil_image = Image.fromarray(image_np, mode="RGBA")
            else:
                raise ValueError(f"Unsupported number of channels ({c})")

            return pil_image
        except Exception as e:
            logger.error(f"Error in tensor_to_pil: {e}")
            raise

    def pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        return image_tensor

    def get_gray_image(self, img: Image.Image) -> np.ndarray:
        img_rgb = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            img_rgb.paste(img, mask=img.split()[3])
        else:
            img_rgb = img.convert("RGB")
        img_gray = img_rgb.convert("L")
        return np.array(img_gray, dtype=np.float32)

    def remove_bg(self, img: Image.Image, tolerance: int) -> Image.Image:
        img = img.convert("RGBA")
        data = np.array(img)

        # Encontrar color más común en las esquinas
        corners = [
            tuple(data[0, 0]),
            tuple(data[0, -1]),
            tuple(data[-1, 0]),
            tuple(data[-1, -1]),
        ]
        bg_color_tuple = Counter(corners).most_common(1)[0][0]
        bg_color = np.array(bg_color_tuple[:3])

        # Si la imagen ya es transparente en las esquinas, no hacer nada extra
        if bg_color_tuple[3] == 0:
            return img

        color_diff = np.abs(data[..., :3].astype(np.int32) - bg_color.astype(np.int32))
        mask = np.all(color_diff <= tolerance, axis=-1)

        # Usar componentes conectados para borrar solo el fondo exterior, no los colores internos parecidos
        structure = np.ones((3, 3), dtype=np.int32)
        labeled_result = label(mask, structure=structure)
        labeled_mask = labeled_result[0]  # type: ignore

        corner_labels = [
            labeled_mask[0, 0],
            labeled_mask[0, -1],
            labeled_mask[-1, 0],
            labeled_mask[-1, -1],
        ]
        valid_labels = set(l for l in corner_labels if l != 0)

        if valid_labels:
            connected_bg_mask = np.isin(labeled_mask, list(valid_labels))
            data[connected_bg_mask, 3] = 0

        return Image.fromarray(data, mode="RGBA")

    def edge_aware_detect(
        self,
        img: Image.Image,
        tile_grid_size: int = 3,
        min_peak_distance: int = 5,
        peak_prominence_factor: float = 0.1,
    ) -> int:
        np_gray = self.get_gray_image(img)
        h, w = np_gray.shape

        tile_h = h // tile_grid_size
        tile_w = w // tile_grid_size

        if tile_h <= 1 or tile_w <= 1:
            informative_tiles = [np_gray]
        else:
            informative_tiles = []
            tile_variances = []
            for i in range(tile_grid_size):
                for j in range(tile_grid_size):
                    y1, y2 = i * tile_h, (i + 1) * tile_h
                    x1, x2 = j * tile_w, (j + 1) * tile_w
                    tile = np_gray[y1:y2, x1:x2]
                    if tile.size > 0:
                        var = np.var(tile)
                        tile_variances.append((var, tile))

            tile_variances.sort(key=lambda x: x[0], reverse=True)
            num_tiles_to_use = max(1, len(tile_variances) // 2)
            informative_tiles = [tile for _, tile in tile_variances[:num_tiles_to_use]]

        all_distances = []

        for idx, tile in enumerate(informative_tiles):
            tile_h, tile_w = tile.shape
            if tile_h < 10 or tile_w < 10:
                continue

            sobel_x = sobel(tile, axis=1, mode="constant")
            sobel_y = sobel(tile, axis=0, mode="constant")

            abs_sobel_x = np.abs(sobel_x)
            abs_sobel_y = np.abs(sobel_y)

            profile_x = np.sum(abs_sobel_x, axis=0)
            profile_y = np.sum(abs_sobel_y, axis=1)

            def find_peaks_simple(profile, min_dist, prom_factor):
                peaks = []
                if len(profile) < 2:
                    return peaks
                max_val = np.max(profile)
                if max_val <= 0:
                    return peaks
                min_prominence = max_val * prom_factor

                for i in range(1, len(profile) - 1):
                    if profile[i] > profile[i - 1] and profile[i] > profile[i + 1]:
                        if profile[i] >= min_prominence:
                            peaks.append(i)

                if len(peaks) <= 1:
                    return peaks
                filtered_peaks = [peaks[0]]
                for i in range(1, len(peaks)):
                    if peaks[i] - filtered_peaks[-1] >= min_dist:
                        filtered_peaks.append(peaks[i])
                return filtered_peaks

            peaks_x = find_peaks_simple(
                profile_x, min_peak_distance, peak_prominence_factor
            )
            peaks_y = find_peaks_simple(
                profile_y, min_peak_distance, peak_prominence_factor
            )

            def calculate_distances(peaks):
                distances = []
                for i in range(1, len(peaks)):
                    distances.append(peaks[i] - peaks[i - 1])
                return distances

            distances_x = calculate_distances(peaks_x)
            distances_y = calculate_distances(peaks_y)

            all_distances.extend(distances_x)
            all_distances.extend(distances_y)

        if not all_distances:
            return 1

        filtered_distances = [d for d in all_distances if d >= 2]
        if not filtered_distances:
            return 1

        distance_counts = Counter(filtered_distances)
        most_common_distances = [dist for dist, _ in distance_counts.most_common(20)]

        if not most_common_distances:
            return 1

        scale_mode = distance_counts.most_common(1)[0][0]

        try:
            if len(most_common_distances) == 1:
                gcd_val = most_common_distances[0]
            else:
                gcd_val = math.gcd(most_common_distances[0], most_common_distances[1])
                for dist in most_common_distances[2:]:
                    gcd_val = math.gcd(gcd_val, dist)
                    if gcd_val == 1:
                        break

            scale_gcd = max(1, gcd_val)

            max_reasonable_scale = min(img.width, img.height) // 4
            if 2 <= scale_gcd <= max_reasonable_scale:
                final_scale = scale_gcd
            elif 2 <= scale_mode <= max_reasonable_scale:
                final_scale = scale_mode
            else:
                final_scale = 1

        except Exception as e:
            max_reasonable_scale = min(img.width, img.height) // 4
            if 2 <= scale_mode <= max_reasonable_scale:
                final_scale = scale_mode
            else:
                final_scale = 1

        return final_scale

    def find_optimal_crop(self, img: Image.Image, scale: int) -> tuple[int, int]:
        if scale <= 1:
            return (0, 0)

        np_gray = self.get_gray_image(img)

        sobel_x = sobel(np_gray, axis=1, mode="constant")
        sobel_y = sobel(np_gray, axis=0, mode="constant")

        abs_sobel_x = np.abs(sobel_x)
        abs_sobel_y = np.abs(sobel_y)

        profile_x = np.sum(abs_sobel_x, axis=0)
        profile_y = np.sum(abs_sobel_y, axis=1)

        def find_best_offset(profile, scale_candidate):
            if scale_candidate <= 1 or len(profile) < scale_candidate:
                return 0

            max_score = -1
            best_offset = 0
            for offset in range(scale_candidate):
                current_score = 0
                idx = offset
                while idx < len(profile):
                    current_score += profile[idx]
                    idx += scale_candidate
                if current_score > max_score:
                    max_score = current_score
                    best_offset = offset
            return best_offset

        best_dx = find_best_offset(profile_x, scale)
        best_dy = find_best_offset(profile_y, scale)

        return (best_dx, best_dy)

    def quantize_image(
        self, img: Image.Image, max_colors: int
    ) -> tuple[Image.Image, list]:
        has_alpha = img.mode == "RGBA"
        alpha = None
        if has_alpha:
            alpha = np.array(img.split()[3])
            alpha = np.where(alpha > 127, 255, 0).astype(np.uint8)
            img_rgb = img.convert("RGB")
        else:
            img_rgb = img.convert("RGB")

        unique_colors = len(set(img_rgb.getdata()))  # type: ignore

        if unique_colors <= max_colors:
            palette = list(set(img_rgb.getdata()))  # type: ignore
            quantized_img = img_rgb
        else:
            np_img = np.array(img_rgb)
            h, w, c = np_img.shape
            data = np_img.reshape((-1, 3))

            kmeans = KMeans(n_clusters=max_colors, n_init="auto", random_state=0).fit(
                data
            )
            labels = kmeans.labels_
            palette_rgb = kmeans.cluster_centers_.round().astype(np.uint8)

            new_data = palette_rgb[labels]
            new_img_np = new_data.reshape((h, w, c))
            quantized_img = Image.fromarray(new_img_np, mode="RGB")
            palette = list(set(quantized_img.getdata()))  # type: ignore

        if has_alpha and alpha is not None:
            quantized_img.putalpha(Image.fromarray(alpha, mode="L"))

        return quantized_img, palette

    def downscale_by_dominant_color(self, img: Image.Image, scale: int) -> Image.Image:
        if scale <= 1:
            return img

        orig_w, orig_h = img.size
        target_w = orig_w // scale
        target_h = orig_h // scale

        if target_w <= 0 or target_h <= 0:
            return img

        img_mode = img.mode
        img_array = np.array(img)
        img_array = img_array[: target_h * scale, : target_w * scale]

        channels = img_array.shape[2] if len(img_array.shape) == 3 else 1

        reshaped = img_array.reshape(target_h, scale, target_w, scale, channels)
        reshaped = reshaped.transpose(0, 2, 1, 3, 4)
        block_view = reshaped.reshape(target_h, target_w, -1, channels)

        downsampled_array = np.zeros((target_h, target_w, channels), dtype=np.uint8)

        for i in range(target_h):
            for j in range(target_w):
                block = block_view[i, j]
                if channels == 4:
                    solid_pixels = block[block[:, 3] > 127]
                    if len(solid_pixels) > 0:
                        unique_colors, counts = np.unique(
                            solid_pixels, axis=0, return_counts=True
                        )
                    else:
                        unique_colors, counts = np.unique(
                            block, axis=0, return_counts=True
                        )
                else:
                    unique_colors, counts = np.unique(block, axis=0, return_counts=True)

                if len(counts) > 0:
                    dominant_idx = np.argmax(counts)
                    downsampled_array[i, j] = unique_colors[dominant_idx]

        return Image.fromarray(downsampled_array, mode=img_mode)

    def jaggy_cleaner(self, img: Image.Image) -> Image.Image:
        has_alpha = img.mode == "RGBA"
        alpha_b = None
        if has_alpha:
            alpha = np.array(img.split()[3])
            alpha_b = np.where(alpha > 127, 255, 0).astype(np.uint8)

        img_rgb = img.convert("RGB")
        np_img = np.array(img_rgb)
        h, w, c = np_img.shape
        out_img = np_img.copy()

        def get_color(x, y):
            if x < 0 or x >= w or y < 0 or y >= h:
                return np.array([0, 0, 0])
            return np_img[y, x]

        def color_distance(c1, c2):
            return np.sum((c1.astype(np.int32) - c2.astype(np.int32)) ** 2)

        threshold_sq = 10000

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if has_alpha and alpha_b is not None and alpha_b[y, x] < 128:
                    continue

                current_color = get_color(x, y)
                neighbors = [
                    get_color(x, y - 1),
                    get_color(x, y + 1),
                    get_color(x - 1, y),
                    get_color(x + 1, y),
                ]

                avg_neighbor_color = np.mean(neighbors, axis=0)
                dist_sq_to_avg = color_distance(current_color, avg_neighbor_color)

                if dist_sq_to_avg > threshold_sq:
                    out_img[y, x] = avg_neighbor_color.astype(np.uint8)

        cleaned_img = Image.fromarray(out_img, mode="RGB")
        if has_alpha and alpha_b is not None:
            cleaned_img.putalpha(Image.fromarray(alpha_b, mode="L"))

        return cleaned_img

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
    ):
        start_time = time.time()
        pil_image = self.tensor_to_pil(image)

        # 0. REMOVE BACKGROUND FIRST
        if remove_background:
            try:
                pil_image = self.remove_bg(pil_image, background_tolerance)
            except Exception as e:
                logger.error(f"Background removal failed: {e}")

        # DETERMINE SCALE
        if target_width > 0 and target_height > 0:
            scale_w = pil_image.width / target_width
            scale_h = pil_image.height / target_height
            # Min scale means it guarantees to fit inside the target dimension
            scale = max(1, round(min(scale_w, scale_h)))
        elif manual_scale_factor > 0:
            scale = manual_scale_factor
        else:
            if scale_detection_method == "edge_aware":
                try:
                    scale = self.edge_aware_detect(
                        pil_image,
                        tile_grid_size=ea_tile_grid_size,
                        min_peak_distance=ea_min_peak_distance,
                        peak_prominence_factor=ea_peak_prominence_factor,
                    )
                    scale = max(1, scale)
                except Exception as e:
                    logger.error(f"Scale detection failed: {e}")
                    scale = 1
            else:
                scale = 1

        crop_x, crop_y = 0, 0
        cropped_pil_image = pil_image
        if scale > 1:
            try:
                crop_x, crop_y = self.find_optimal_crop(pil_image, scale)
                new_width = ((pil_image.width - crop_x) // scale) * scale
                new_height = ((pil_image.height - crop_y) // scale) * scale

                if new_width > 0 and new_height > 0:
                    box = (crop_x, crop_y, crop_x + new_width, crop_y + new_height)
                    cropped_pil_image = pil_image.crop(box)
            except Exception as e:
                logger.error(f"Optimal crop failed: {e}")

        # 1. DOWNSCALE FIRST
        final_image = cropped_pil_image
        if scale > 1:
            try:
                if downscale_method == "dominant":
                    final_image = self.downscale_by_dominant_color(final_image, scale)
                elif downscale_method == "nearest":
                    target_w_scaled = max(1, final_image.width // scale)
                    target_h_scaled = max(1, final_image.height // scale)
                    final_image = final_image.resize(
                        (target_w_scaled, target_h_scaled), Image.Resampling.NEAREST
                    )
                else:
                    final_image = self.downscale_by_dominant_color(final_image, scale)
            except Exception as e:
                logger.error(f"Downscaling failed: {e}")

        # OBLIGAR A QUE LA IMAGEN REESCALADA ENCAJE EXACTAMENTE ANTES DE HACER PADDING
        if target_width > 0 and target_height > 0:
            final_image = final_image.resize(
                (target_width, target_height), Image.Resampling.NEAREST
            )

        # 2. EXACT TARGET SIZE PADDING/CROPPING (Esto ya no debería ser necesario si forzamos el resize exacto arriba, pero lo mantenemos por seguridad)
        if target_width > 0 and target_height > 0:
            final_exact = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))

            # Si la imagen recortada y escalada es más grande que el target, la recortamos desde el centro
            if final_image.width > target_width or final_image.height > target_height:
                left = max(0, (final_image.width - target_width) // 2)
                top = max(0, (final_image.height - target_height) // 2)
                right = min(final_image.width, left + target_width)
                bottom = min(final_image.height, top + target_height)
                final_image = final_image.crop((left, top, right, bottom))

            # Pegarla en el centro (si es más pequeña, se le añade padding transparente)
            offset_x = max(0, (target_width - final_image.width) // 2)
            offset_y = max(0, (target_height - final_image.height) // 2)

            final_exact.paste(final_image, (offset_x, offset_y))
            final_image = final_exact

        # 3. QUANTIZE
        try:
            final_image, _ = self.quantize_image(final_image, max_colors)
        except Exception as e:
            logger.error(f"Quantization failed: {e}")

        # 4. CLEANUP
        if cleanup_jaggies and scale > 1:
            try:
                final_image = self.jaggy_cleaner(final_image)
            except Exception as e:
                logger.error(f"Jaggy cleanup failed: {e}")

        processing_time_ms = round((time.time() - start_time) * 1000)
        manifest = str(
            {
                "detected_scale": scale,
                "crop_offset": (crop_x, crop_y),
                "processing_time_ms": processing_time_ms,
            }
        )

        output_tensor = self.pil_to_tensor(final_image)

        # Crear versión ampliada para previsualización usando Nearest Neighbor
        preview_width = max(1, final_image.width * upscale_preview_factor)
        preview_height = max(1, final_image.height * upscale_preview_factor)
        preview_image = final_image.resize(
            (preview_width, preview_height), Image.Resampling.NEAREST
        )

        # Guardar en la carpeta temporal de ComfyUI para la previsualización del nodo
        temp_dir = folder_paths.get_temp_directory()
        filename_prefix = "unfake_preview"
        filename = f"{filename_prefix}_{random.randint(100000, 999999)}.png"
        preview_path = os.path.join(temp_dir, filename)
        preview_image.save(preview_path, format="PNG")

        results = [{"filename": filename, "subfolder": "", "type": "temp"}]

        return {"ui": {"images": results}, "result": (output_tensor, manifest)}


NODE_CLASS_MAPPINGS = {
    "UnfakePixelArtNode": UnfakePixelArtNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnfakePixelArtNode": "Unfake Pixel Art (AI to Real)",
}
