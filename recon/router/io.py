"""Unchanged codec/ADM image loading functions, configured by the caller."""
import hashlib
import re
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], crop):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Lambda(lambda image: crop(image, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))

def load_codec(args, device):
    source = args.reconstruction_root.resolve()
    sys.path.insert(0, str(source))
    from models import TiTokLlamaGenStage2
    from train_refinement import DynamicBudgetRouter
    from train_base import add_path, load_llamagen_vq, load_titok

    add_path(str(args.llamagen_root.resolve()))
    from dataset.augmentation import center_crop_arr

    titok = load_titok(str(args.titok_root), str(args.titok_config),
                       str(args.titok_ckpt), device)
    vq = load_llamagen_vq(str(args.llamagen_root), str(args.llamagen_ckpt),
                          device, 16384, 8)
    model = TiTokLlamaGenStage2(
        titok, vq, lg_latent_channels=256, head_channels=256,
        head_mode="feature", codebook_size=16384,
        codebook_temperature=1.0,
    ).to(device)
    model.router = DynamicBudgetRouter(
        latent_channels=256, hidden_dim=128, depth=3,
        target_ratio=0.375, min_ratio=0.0, max_ratio=1.0,
        detach_inputs=True,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_ema" not in checkpoint:
        raise RuntimeError("reconstruction checkpoint has no model_ema")
    state = checkpoint["model_ema"]
    params = dict(model.named_parameters())
    unknown = set(state) - set(params)
    missing_latent = {name for name in params if name.startswith("latent_decoder.")} - set(state)
    if unknown or missing_latent:
        raise RuntimeError(f"checkpoint mismatch: unknown={sorted(unknown)[:10]}, missing={sorted(missing_latent)[:10]}")
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value.to(device=params[name].device, dtype=params[name].dtype))
    copied = len(state)
    step = int(checkpoint.get("step", -1))
    epoch = int(checkpoint.get("epoch", -1))
    match = re.search(r"epoch_(\d+)_step_(\d+)\.pt$", Path(args.checkpoint).name)
    if match and (step >= 0 and step != int(match.group(2)) or epoch >= 0 and epoch != int(match.group(1))):
        raise RuntimeError("checkpoint filename and metadata disagree")
    if copied != 692:
        raise RuntimeError(f"Expected 692 released EMA parameters, got {copied}")
    model.eval().requires_grad_(False)
    def native_llamagen_feature(vq_model, image, codebook_embed_dim, allow_encoder_grad=False):
        if allow_encoder_grad:
            quant, _, info = vq_model.encode(image)
        else:
            with torch.no_grad():
                quant, _, info = vq_model.encode(image)
                quant = quant.detach()
        indices = None
        if info is not None and len(info) >= 3 and info[2] is not None:
            indices = info[2].view(image.shape[0], quant.shape[-2], quant.shape[-1]).long()
        if quant.ndim != 4 or quant.shape[-2:] != (16, 16):
            raise ValueError(f"unexpected LlamaGen encode output shape: {tuple(quant.shape)}")
        if quant.shape[1] == codebook_embed_dim:
            return vq_model.post_quant_conv(quant), indices
        if quant.shape[1] == 256:
            return quant, indices
        raise ValueError(f"unexpected LlamaGen encode channels: {tuple(quant.shape)}")

    def to_uint8(image, image_range):
        image01 = image.add(1).mul(.5) if image_range == 'minus1_1' else image
        return image01.clamp(0, 1).mul(255).round().to(torch.uint8)

    def psnr_sum(prediction, target, image_range):
        if image_range == 'minus1_1':
            prediction, target = prediction.add(1).mul(.5), target.add(1).mul(.5)
        error = (prediction.clamp(0, 1) - target.clamp(0, 1)).square().flatten(1).mean(1)
        return (-10 * torch.log10(error + 1e-8)).sum()

    helpers = SimpleNamespace(native_llamagen_feature=native_llamagen_feature,
                              to_uint8=to_uint8, psnr_sum=psnr_sum)
    return model, helpers, center_crop_arr, {"step": step, "epoch": epoch, "copied": copied}

def feature_tensor(extractor, image_uint8: torch.Tensor) -> np.ndarray:
    features = extractor(image_uint8)
    if isinstance(features, (list, tuple)):
        features = features[0]
    result = features.detach().float().cpu().numpy()
    if result.ndim != 2 or result.shape[1] != 2048 or not np.isfinite(result).all():
        raise ValueError(f"Unexpected Inception features: {result.shape}")
    return result
