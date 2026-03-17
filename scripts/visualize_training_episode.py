"""Render one training episode from a LeRobot v2.0 parquet dataset as an MP4.

Usage:
    python scripts/visualize_training_episode.py \
        --data_dir /home/user/cyn_ws/results/lerobot_dataset \
        --episode 0 \
        --output training_episode_0.mp4
"""

import argparse
import io
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
from PIL import Image


def decode_image(img_cell) -> np.ndarray:
    if isinstance(img_cell, dict):
        raw = img_cell.get("bytes", img_cell.get("data"))
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.asarray(pil)
    elif isinstance(img_cell, np.ndarray):
        return img_cell
    else:
        raise TypeError(f"Unexpected image type: {type(img_cell)}")


def main():
    parser = argparse.ArgumentParser(description="Visualize a training episode")
    parser.add_argument("--data_dir", type=str,
                        default="/home/user/cyn_ws/results/lerobot_dataset")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--image_keys", type=str, nargs="+",
                        default=["image", "extra_image_0", "extra_image_1"])
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ep_idx = args.episode
    chunk_idx = ep_idx // 1000
    parquet_path = data_dir / f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"Episode {ep_idx}: {len(df)} frames, columns: {list(df.columns)}")

    available_keys = [k for k in args.image_keys if k in df.columns]
    if not available_keys:
        raise ValueError(f"No image columns found. Available: {list(df.columns)}")
    print(f"Using image keys: {available_keys}")

    output_path = args.output or f"training_episode_{ep_idx}.mp4"

    writer = imageio.get_writer(output_path, fps=args.fps)
    for row_idx in range(len(df)):
        row = df.iloc[row_idx]
        views = []
        for key in available_keys:
            img = decode_image(row[key])
            views.append(img)
        frame = np.concatenate(views, axis=1)
        writer.append_data(frame)
    writer.close()

    print(f"Saved {len(df)}-frame video to {output_path}  "
          f"({frame.shape[1]}x{frame.shape[0]}, {args.fps} fps)")


if __name__ == "__main__":
    main()
