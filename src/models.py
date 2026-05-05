from __future__ import annotations

import time
from typing import Any

import timm
import torch
from torch import nn
from timm.data import create_transform, resolve_model_data_config


MODEL_REGISTRY: dict[str, str] = {
    "dinov3_convnext_tiny": "convnext_tiny.dinov3_lvd1689m",
    "dinov3_vit_s_plus": "vit_small_plus_patch16_dinov3.lvd1689m",
    "dinov3_convnext_base": "convnext_base.dinov3_lvd1689m",
    "dinov3_vit_b": "vit_base_patch16_dinov3.lvd1689m",
    "dinov3_convnext_large": "convnext_large.dinov3_lvd1689m",
    "dinov3_vit_l": "vit_large_patch16_dinov3.lvd1689m",
    "resnet50": "resnet50.a1_in1k",
}

HEAD_KEYS = ("head", "fc", "classifier")
FEATURE_POOLS = {"default", "cls", "patch_mean", "cls_patch_mean"}


class FeaturePoolClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, feature_pool: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_pool = feature_pool
        self.num_features = _pooled_dim(backbone, feature_pool)
        self.head = nn.Linear(self.num_features, num_classes)
        self.default_cfg = getattr(backbone, "default_cfg", None)
        self.pretrained_cfg = getattr(backbone, "pretrained_cfg", None)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return extract_features(self.backbone, x, self.feature_pool)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def get_classifier(self) -> nn.Module:
        return self.head


def create_classifier(model_name: str, num_classes: int, pretrained: bool = True, feature_pool: str = "default") -> torch.nn.Module:
    timm_name = MODEL_REGISTRY.get(model_name, model_name)
    if feature_pool not in FEATURE_POOLS:
        raise ValueError(f"Unknown feature_pool: {feature_pool}")
    if feature_pool == "default":
        return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
    return FeaturePoolClassifier(backbone, num_classes, feature_pool)


def build_transforms(model: torch.nn.Module, train: bool, image_size: int | None = None) -> Any:
    cfg = resolve_model_data_config(getattr(model, "backbone", model))
    if image_size is not None:
        cfg["input_size"] = (3, int(image_size), int(image_size))
    return create_transform(**cfg, is_training=train)


def extract_features(model: nn.Module, x: torch.Tensor, feature_pool: str = "default") -> torch.Tensor:
    if feature_pool == "default":
        return model.forward_head(model.forward_features(x), pre_logits=True)
    cls, patches = _vit_tokens(model, x)
    if feature_pool == "cls":
        return cls
    patch_mean = patches.mean(dim=1)
    if feature_pool == "patch_mean":
        return patch_mean
    if feature_pool == "cls_patch_mean":
        return torch.cat([cls, patch_mean], dim=1)
    raise ValueError(f"Unknown feature_pool: {feature_pool}")


def _vit_tokens(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = model.forward_features(x)
    if isinstance(tokens, dict):
        if "x_norm_clstoken" in tokens and "x_norm_patchtokens" in tokens:
            return tokens["x_norm_clstoken"], tokens["x_norm_patchtokens"]
        tokens = tokens.get("x", tokens.get("last_hidden_state"))
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        raise RuntimeError("CLS/patch pooling requires ViT token features")
    prefix = int(getattr(model, "num_prefix_tokens", 1) or 1)
    cls, patches = tokens[:, 0], tokens[:, prefix:]
    if patches.shape[1] == 0:
        raise RuntimeError("No patch tokens found after CLS/register tokens")
    return cls, patches


def _pooled_dim(model: nn.Module, feature_pool: str) -> int:
    dim = int(getattr(model, "num_features", 0))
    if dim <= 0:
        raise RuntimeError("Backbone does not expose num_features")
    return dim * 2 if feature_pool == "cls_patch_mean" else dim


def configure_trainable(model: torch.nn.Module, mode: str, last_blocks: int = 2) -> None:
    for p in model.parameters():
        p.requires_grad = mode == "full_finetune"
    if mode == "linear_probe":
        _unfreeze_classifier(model)
    elif mode == "partial_finetune":
        _unfreeze_classifier(model)
        _unfreeze_last_blocks(model, last_blocks)
    elif mode != "full_finetune":
        raise ValueError(f"Unknown training mode: {mode}")


def _unfreeze_classifier(model: torch.nn.Module) -> None:
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if isinstance(classifier, torch.nn.Module):
        for p in classifier.parameters():
            p.requires_grad = True
        return
    for name, p in model.named_parameters():
        if name.split(".", 1)[0].lower() in HEAD_KEYS:
            p.requires_grad = True


def _unfreeze_last_blocks(model: torch.nn.Module, n: int) -> None:
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        _unfreeze_last_blocks(backbone, n)
        return
    stages = getattr(model, "stages", None)
    if stages is not None:
        blocks = []
        for stage in stages:
            stage_blocks = getattr(stage, "blocks", None)
            if stage_blocks is not None:
                blocks.extend(list(stage_blocks))
        if blocks:
            for block in blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad = True
            return
    for attr in ("blocks", "layers", "layer4", "layer3"):
        blocks = getattr(model, attr, None)
        if blocks is not None:
            items = list(blocks) if hasattr(blocks, "__iter__") else [blocks]
            for block in items[-n:]:
                for p in block.parameters():
                    p.requires_grad = True
            return
    for child in list(model.children())[-n:]:
        for p in child.parameters():
            p.requires_grad = True


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def freeze_frozen_batchnorm_stats(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


def profile_model(model: torch.nn.Module, image_size: tuple[int, int, int], device: torch.device) -> dict[str, float | None]:
    model.eval().to(device)
    x = torch.randn(1, *image_size, device=device)
    params = sum(p.numel() for p in model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(5):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        reps = 20
        for _ in range(reps):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    latency = (time.perf_counter() - start) / reps
    vram = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
    return {
        "params": float(params),
        "latency_batch1_sec": latency,
        "throughput_img_sec": 1.0 / latency,
        "peak_vram_mb": vram,
    }
