"""Contains `sharp predict` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F

from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    save_ply,
    unproject_gaussians,
)

from .render import render_gaussians

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


@click.command()
@click.option("-i", "--input-path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("-o", "--output-path", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("-c", "--checkpoint-path", type=click.Path(path_type=Path, dir_okay=False), default=None)
@click.option("--render/--no-render", "with_rendering", is_flag=True, default=False)
@click.option("--device", type=str, default="default")
@click.option("--decimate", type=int, default=1, help="Reduce splat count (1=Keep All, 2=Half, 4=Quarter).")
@click.option("-v", "--verbose", is_flag=True)
def predict_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    with_rendering: bool,
    device: str,
    decimate: int,
    verbose: bool,
):
    """Predict Gaussians from input images."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    # ... Image Loading Boilerplate ...
    extensions = io.get_supported_image_extensions()
    image_paths = [input_path] if input_path.is_file() else []
    if not input_path.is_file():
        for ext in extensions:
            image_paths.extend(list(input_path.glob(f"**/*{ext}")))

    if not image_paths:
        LOGGER.info("No valid images found.")
        return

    if device == "default":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    LOGGER.info("Using device %s", device)

    # Load Model
    if checkpoint_path is None:
        LOGGER.info("Downloading default model...")
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)

    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)

    output_path.mkdir(exist_ok=True, parents=True)

    for image_path in image_paths:
        LOGGER.info(f"Processing {image_path.name}")
        image, _, f_px = io.load_rgb(image_path)
        height, width = image.shape[:2]

        intrinsics = torch.tensor(
            [[f_px, 0, (width - 1) / 2.0, 0], [0, f_px, (height - 1) / 2.0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            device=device, dtype=torch.float32
        )
        
        # 1. Run Original Inference
        gaussians = predict_image(gaussian_predictor, image, f_px, torch.device(device))

        # === 2. DECIMATION (The only change) ===
        if decimate > 1:
            LOGGER.info(f"Decimating {decimate}x (Original: {gaussians.mean_vectors.shape[1]} splats)")
            
            # Slicing with [:, ::decimate, :] preserves the Batch dimension [1, N, C].
            # This keeps save_ply happy.
            gaussians = Gaussians3D(
                mean_vectors=gaussians.mean_vectors[:, ::decimate, :],
                singular_values=gaussians.singular_values[:, ::decimate, :],
                quaternions=gaussians.quaternions[:, ::decimate, :],
                colors=gaussians.colors[:, ::decimate, :],
                opacities=gaussians.opacities[:, ::decimate] # Opacities are [1, N]
            )

        LOGGER.info(f"Saving {gaussians.mean_vectors.shape[1]} splats to {output_path}")
        save_ply(gaussians, f_px, (height, width), output_path / f"{image_path.stem}.ply")

        if with_rendering:
             metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")
             render_gaussians(gaussians, metadata, (output_path / image_path.stem).with_suffix(".mp4"))


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
) -> Gaussians3D:
    """Predict Gaussians (Original Implementation)."""
    # Original settings
    internal_shape = (1536, 1536) 

    LOGGER.info("Running preprocessing.")
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    # Standard Stretch (Original Repo Behavior)
    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    LOGGER.info("Running inference.")
    gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    LOGGER.info("Running postprocessing.")
    intrinsics = torch.tensor(
        [[f_px, 0, width / 2, 0], [0, f_px, height / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    ).float().to(device)
    
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Standard Unprojection (Original Repo Behavior)
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    return gaussians