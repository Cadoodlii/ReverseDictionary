#!/usr/bin/env python3
"""Small example script to demonstrate loading the PseudoRDMIF dataset
and saving a few sample images with their labels & definitions.

Usage:
  python3 examples/test_dataset.py --n 3 [--show]

The script writes saved images to `examples/output/` and prints sample info.
"""
import argparse
import os
from pathlib import Path
import textwrap

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
except Exception:
    torch = None

from dataset import PseudoRDMIFDataset


def tensor_to_pil(image_tensor):
    """Convert a torchvision-style tensor or PIL image to a PIL Image.
    Accepts either a PIL.Image.Image or a torch.Tensor (C,H,W).
    """
    try:
        from PIL import Image
    except Exception:
        Image = None

    if Image is None:
        raise RuntimeError("Pillow not available; install pillow to convert images.")

    # If already a PIL image
    if hasattr(image_tensor, "__class__") and image_tensor.__class__.__name__ == "Image":
        return image_tensor

    # Torch tensor
    if torch is not None and isinstance(image_tensor, torch.Tensor):
        img = image_tensor.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.transpose(1, 2, 0)
        # If float in 0..1, scale to 0..255
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        from PIL import Image as PILImage

        if img.ndim == 2:
            return PILImage.fromarray(img)
        return PILImage.fromarray(img)

    # As a last resort try numpy array
    if isinstance(image_tensor, np.ndarray):
        from PIL import Image as PILImage

        arr = image_tensor
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        return PILImage.fromarray(arr.astype('uint8'))

    raise TypeError("Unsupported image type: %r" % type(image_tensor))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="dataset.json", help="Path to dataset.json")
    parser.add_argument("--n", type=int, default=3, help="Number of samples to save")
    parser.add_argument("--show", action="store_true", help="Also display images with matplotlib (may fail on headless systems)")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = PseudoRDMIFDataset(args.json)
    n = min(args.n, len(ds))
    print(f"Loaded dataset with {len(ds)} items; saving {n} samples to {out_dir!s}")

    for i in range(n):
        sample = ds[i]
        word = sample.get("word")
        definition = sample.get("definition")
        img_obj = sample.get("image")
        image_path = sample.get("image_path")

        try:
            pil = tensor_to_pil(img_obj)
        except Exception as e:
            print(f"Could not convert sample {i} image to PIL: {e}")
            continue

        fname = out_dir / f"sample_{i}_{sample.get('synset_id','unknown')}.png"
        pil.save(fname)

        print("Sample", i)
        print(" - word:", word)
        print(" - definition:", textwrap.shorten(definition, width=200))
        print(" - saved image:", fname)
        print(" - original image file:", image_path)

        if args.show:
            plt.figure(figsize=(4, 4))
            plt.imshow(pil)
            plt.axis('off')
            plt.title(f"{word}: {textwrap.shorten(definition, width=60)}")
            plt.show()


if __name__ == "__main__":
    main()
