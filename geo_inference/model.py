import numpy as np
import torch
import torch.nn.functional as F
from rasterio.windows import Window
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def mps_available():
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


def resolve_device(setting="auto"):
    setting = (setting or "auto").lower()
    if setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if setting.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Device {setting!r} requested but CUDA is not available")
    if setting == "mps" and not mps_available():
        raise RuntimeError("Device 'mps' requested but Apple GPU (MPS) is not available")
    return torch.device(setting)


def describe_device(device):
    if device.type == "cuda":
        return f"GPU: {torch.cuda.get_device_name(device)}"
    if device.type == "mps":
        return "GPU: Apple Silicon (MPS)"
    return "CPU (no hardware acceleration)"


def load_model(model_dir, device):
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"{model_dir} has no config.json; use the saved best_model directory.")
    processor = SegformerImageProcessor.from_pretrained(model_dir, do_resize=False)
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(device)
    model.eval()
    return model, processor

def starts(length, tile, stride):
    if length <= tile:
        return [0]
    vals = list(range(0, length - tile + 1, stride))
    if vals[-1] != length - tile:
        vals.append(length - tile)
    return vals

def to_rgb(data):
    if data.shape[0] >= 3:
        arr = np.moveaxis(data[:3], 0, -1)
    elif data.shape[0] == 1:
        arr = np.repeat(data[0][..., None], 3, axis=-1)
    else:
        raise ValueError(f"Unsupported bands: {data.shape[0]}")
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    result = np.zeros(arr.shape, dtype=np.uint8)
    for c in range(3):
        b = arr[..., c]
        finite = np.isfinite(b)
        if finite.any():
            lo, hi = np.percentile(b[finite], [1, 99])
            if hi <= lo:
                hi = lo + 1
            result[..., c] = np.clip((b - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
    return result

def tensor_for(processor, rgb):
    enc = processor(images=rgb, return_tensors="pt", do_resize=False,
                    do_rescale=True, do_normalize=True)
    return enc["pixel_values"].squeeze(0)

def predict_batch(model, pixel_values, tile_size, device):
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        logits = model(pixel_values=pixel_values.to(device)).logits
        logits = F.interpolate(logits, size=(tile_size, tile_size),
                               mode="bilinear", align_corners=False)
    return torch.sigmoid(logits[:, 0]).detach().float().cpu().numpy()

def iter_predict_raster(src, model, processor, args, device):
    h, w = src.height, src.width
    tile, stride = args.tile_size, args.tile_size - args.overlap
    if stride <= 0:
        raise ValueError("--overlap must be smaller than --tile-size")
    coords = [(x, y, min(tile, w-x), min(tile, h-y))
              for y in starts(h, tile, stride) for x in starts(w, tile, stride)]
    prob_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    one = np.maximum(np.hanning(tile + 2)[1:-1].astype(np.float32), 0.05)
    weights = np.outer(one, one)
    print(f"  {len(coords)} tiles to infer")
    finalized = 0
    for off in range(0, len(coords), args.batch_size):
        batch = coords[off:off + args.batch_size]
        tensors = []
        for x, y, ww, hh in batch:
            rgb = to_rgb(src.read(window=Window(x, y, ww, hh)))
            if ww != tile or hh != tile:
                pad = np.zeros((tile, tile, 3), dtype=np.uint8)
                pad[:hh, :ww] = rgb
                rgb = pad
            tensors.append(tensor_for(processor, rgb))
        probs = predict_batch(model, torch.stack(tensors), tile, device)
        for prob, (x, y, ww, hh) in zip(probs, batch):
            wt = weights[:hh, :ww]
            prob_sum[y:y+hh, x:x+ww] += prob[:hh, :ww] * wt
            weight_sum[y:y+hh, x:x+ww] += wt
        done = min(off + len(batch), len(coords))
        print(f"\r  Inference {done}/{len(coords)}", end="", flush=True)
        final_y = coords[done][1] if done < len(coords) else h
        if final_y > finalized:
            rows = prob_sum[finalized:final_y] / np.maximum(weight_sum[finalized:final_y], 1e-8)
            yield finalized, final_y, rows
            finalized = final_y
    print()


def predict_raster(src, model, processor, args, device):
    prob = np.zeros((src.height, src.width), dtype=np.float32)
    for y0, y1, rows in iter_predict_raster(src, model, processor, args, device):
        prob[y0:y1] = rows
    return prob
