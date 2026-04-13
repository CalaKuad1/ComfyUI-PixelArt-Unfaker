# ComfyUI Pixel Art Unfaker

A powerful custom node for ComfyUI that transforms "fake" or blurry AI-generated pixel art into true, crisp, mathematically perfect pixel art. 

Based on the brilliant algorithm from [unfake.js](https://github.com/jenissimo/unfake.js) and inspired by the early adaptation [ComfyUI-Unfake-Pixels](https://github.com/tauraloke/ComfyUI-Unfake-Pixels).

## 🌟 Features

* **Edge-Aware Scale Detection**: Automatically detects the true "pixel size" of an AI-generated image, even if it's blurry or upscaled.
* **Optimal Crop & Center**: Aligns and crops the image perfectly to the pixel grid. No more half-pixels!
* **Smart Background Removal**: Automatically detects and makes the background transparent based on corner colors using connected components.
* **Exact Target Resolution**: Force the output to be exactly `16x16`, `64x64`, etc. The node will calculate the math, crop, and pad automatically.
* **Intelligent Color Quantization**: Reduces the color palette (e.g., 16 colors) using K-Means clustering for that authentic retro look.
* **Built-in Upscaled Preview**: Generates a crisp, Nearest-Neighbor upscaled preview directly inside the ComfyUI node interface.

## 🛠️ Installation

1. Clone or download this repository into your ComfyUI `custom_nodes` folder:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/TU-USUARIO/ComfyUI-PixelArt-Unfaker.git
```
2. Install the required Python packages:
```bash
cd ComfyUI-PixelArt-Unfaker
pip install -r requirements.txt
```
*(If you use the portable Windows version, use `.\python_embeded\python.exe -m pip install -r ...`)*

3. Restart ComfyUI. You will find the node under the `image/postprocessing` category as **Unfake Pixel Art (AI to Real)**.

## 🎛️ Node Parameters

* `target_width` & `target_height`: The exact final resolution you want (e.g., 64). Set to `0` to keep the auto-detected original aspect ratio.
* `manual_scale_factor`: If auto-detection fails (common on noisy textures like dirt/ores), set this manually (e.g., if input is 1024x1024 and you want 16x16, the scale is 64). Set to `0` for auto-detect.
* `remove_background`: Attempts to make the background transparent automatically.
* `background_tolerance`: Tolerance (0-255) for the background removal color matching.
* `max_colors`: Number of colors to quantize the final image to (e.g., 8, 16, 32).
* `cleanup_jaggies`: Cleans up isolated noise pixels for cleaner lines.
* `downscale_method`: `nearest` (crisper) or `dominant` (smooth color averaging).
* `upscale_preview_factor`: How much to scale up the preview inside the ComfyUI node (e.g., 8x).

## 🙏 Credits & Acknowledgements

This project stands on the shoulders of giants. Huge thanks to:

* **[jenissimo](https://github.com/jenissimo)** for creating the original **[unfake.js](https://github.com/jenissimo/unfake.js)** library and the mathematical algorithm to tame AI pixel art.
* **[tauraloke](https://github.com/tauraloke)** for the **[ComfyUI-Unfake-Pixels](https://github.com/tauraloke/ComfyUI-Unfake-Pixels)** implementation which served as the foundational inspiration for bringing this into ComfyUI.

## 📄 License

MIT License