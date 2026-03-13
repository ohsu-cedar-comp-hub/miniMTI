"""
Extract a small example dataset (~200 cells) from a CRC-Orion sample.

The output HDF5 has the same format as training data:
  - images: (N, 32, 32, 20) — 17 IF channels + 3 H&E channels
  - masks:  (N, 32, 32) — binary cell masks
  - metadata: (N,) — string with cell ID and coordinates

Usage:
    python scripts/create_example_data.py \
        --input /path/to/orion_crc_dataset_sid=CRC05.h5 \
        --output example_data.h5 \
        --num-cells 200

To upload to HuggingFace:
    huggingface-cli upload changlab/miniMTI-CRC-example example_data.h5
"""
import argparse
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Create example dataset for inference demos')
    parser.add_argument('--input', type=str, required=True, help='Path to source HDF5 file')
    parser.add_argument('--output', type=str, default='example_data.h5', help='Output HDF5 file')
    parser.add_argument('--num-cells', type=int, default=200, help='Number of cells to extract')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Reading from {args.input}...")
    with h5py.File(args.input, 'r') as f:
        total = len(f['images'])
        print(f"Total cells: {total}")
        print(f"Image shape: {f['images'].shape}")
        print(f"Mask shape: {f['masks'].shape}")

        n = min(args.num_cells, total)
        idx = np.sort(np.random.choice(total, n, replace=False))

        print(f"Extracting {n} cells...")
        images = f['images'][idx]
        masks = f['masks'][idx]
        metadata = f['metadata'][idx]

    print(f"Writing to {args.output}...")
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('images', data=images, compression='gzip', compression_opts=4)
        f.create_dataset('masks', data=masks, compression='gzip', compression_opts=4)
        f.create_dataset('metadata', data=metadata)

    # Report file size
    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Created {args.output} ({size_mb:.1f} MB, {n} cells)")


if __name__ == "__main__":
    main()
