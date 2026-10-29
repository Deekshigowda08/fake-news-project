"""Image preprocessing, validation, and security sanitization module."""

from pathlib import Path
from typing import Optional, Tuple, Union
import io
import logging
from PIL import Image
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

# Standard Vision Transformer / ImageNet normalization values
VIT_MEAN = [0.485, 0.456, 0.406]
VIT_STD = [0.229, 0.224, 0.225]
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB limit
ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "BMP", "GIF"}


def get_vit_transform(image_size: int = 224) -> transforms.Compose:
    """Returns standard torchvision preprocessing transform for Vision Transformer."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
    ])


class ImageValidationError(Exception):
    """Raised when an image fails format, size, or integrity checks."""
    pass


def validate_and_load_image(
    image_source: Union[str, Path, bytes, io.BytesIO],
    max_size_bytes: int = MAX_IMAGE_BYTES,
) -> Image.Image:
    """Validates image integrity, format, and size, returning a clean PIL RGB Image.
    
    Security safeguards:
    - Rejects files exceeding max_size_bytes.
    - Verifies actual image header and decodability rather than trusting extension.
    - Converts to RGB to strip non-standard channels or embedded metadata payloads.
    """
    data_bytes: bytes

    if isinstance(image_source, Image.Image):
        rgb_image = image_source.convert("RGB")
        rgb_image.load()
        return rgb_image

    if isinstance(image_source, (str, Path)):
        p = Path(image_source)
        if not p.is_file():
            raise FileNotFoundError(f"Image file does not exist: {p}")
        file_size = p.stat().st_size
        if file_size > max_size_bytes:
            raise ImageValidationError(f"Image size {file_size} bytes exceeds limit {max_size_bytes}")
        with open(p, "rb") as f:
            data_bytes = f.read()
    elif isinstance(image_source, bytes):
        if len(image_source) > max_size_bytes:
            raise ImageValidationError(f"Image size {len(image_source)} bytes exceeds limit {max_size_bytes}")
        data_bytes = image_source
    elif isinstance(image_source, io.BytesIO):
        data_bytes = image_source.getvalue()
        if len(data_bytes) > max_size_bytes:
            raise ImageValidationError("Image buffer exceeds maximum permitted size")
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")

    try:
        # First pass: verify integrity
        with Image.open(io.BytesIO(data_bytes)) as test_img:
            img_format = (test_img.format or "").upper()
            if img_format not in ALLOWED_FORMATS:
                raise ImageValidationError(f"Unsupported image format: '{img_format}'. Allowed: {ALLOWED_FORMATS}")
            test_img.verify()

        # Second pass: reopen for actual loading and conversion
        with Image.open(io.BytesIO(data_bytes)) as loaded_img:
            # Convert to RGB (handles RGBA, Palette, Grayscale)
            rgb_image = loaded_img.convert("RGB")
            # Force load image data into memory before closing buffer
            rgb_image.load()
            return rgb_image

    except ImageValidationError:
        raise
    except Exception as e:
        raise ImageValidationError(f"Corrupted or invalid image: {e}") from e


def preprocess_image_for_vit(
    image_source: Optional[Union[str, Path, bytes, io.BytesIO]],
    transform: Optional[transforms.Compose] = None,
) -> Tuple[Optional[torch.Tensor], bool, Optional[str]]:
    """Loads, validates, and transforms image into ViT tensor.
    
    Returns:
        (tensor or None, has_image_bool, error_message or None)
    """
    if image_source is None:
        return None, False, "No image provided"

    transform = transform or get_vit_transform()

    try:
        pil_img = validate_and_load_image(image_source)
        tensor = transform(pil_img)
        return tensor, True, None
    except Exception as e:
        logger.warning("Image preprocessing failed: %s", e)
        return None, False, str(e)
