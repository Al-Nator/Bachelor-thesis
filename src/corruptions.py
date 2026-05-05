from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
import warnings

import numpy as np
from PIL import Image

CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


@dataclass(frozen=True)
class Corruption:
    name: str
    severity: int
    seed: int = 42

    def __call__(self, image: Image.Image, path: str = "") -> Image.Image:
        salt = int.from_bytes(blake2b(path.encode(), digest_size=4).digest(), "little")
        return apply_corruption(image, self.name, self.severity, self.seed + salt)


def apply_corruption(image: Image.Image, name: str, severity: int, seed: int) -> Image.Image:
    if name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption: {name}")
    _numpy_compat()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")
            from imagecorruptions import corrupt
            import imagecorruptions.corruptions as ic
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Install corruptions backend with `uv sync`") from e
    _imagecorruptions_compat(ic)
    severity = int(np.clip(severity, 1, 5))
    image = _min_size_rgb(image, 32)
    state = np.random.get_state()
    try:
        np.random.seed(seed % (2**32))
        out = corrupt(np.asarray(image), corruption_name=name, severity=severity)
    finally:
        np.random.set_state(state)
    return Image.fromarray(out).convert("RGB")


def _min_size_rgb(image: Image.Image, min_size: int) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    if w >= min_size and h >= min_size:
        return image
    scale = max(min_size / max(w, 1), min_size / max(h, 1))
    size = (max(min_size, round(w * scale)), max(min_size, round(h * scale)))
    return image.resize(size, Image.Resampling.BICUBIC)


def _numpy_compat() -> None:
    for alias, target in {"float_": np.float64, "int_": np.int64}.items():
        if not hasattr(np, alias):
            setattr(np, alias, target)


def _imagecorruptions_compat(ic) -> None:
    if getattr(ic.gaussian, "_vkr_compat", False):
        return

    def gaussian(image, sigma=1, output=None, mode="nearest", cval=0, multichannel=None, channel_axis=None, **kwargs):
        from skimage.filters import gaussian as sk_gaussian

        if channel_axis is None and multichannel:
            channel_axis = -1
        return sk_gaussian(image, sigma=sigma, mode=mode, cval=cval, channel_axis=channel_axis, **kwargs)

    gaussian._vkr_compat = True
    ic.gaussian = gaussian
