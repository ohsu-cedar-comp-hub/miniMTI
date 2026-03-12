import os
import gc
import math
import argparse
import h5py
import zarr
import tifffile
import numpy as np
import dask.array as da
from skimage.io import imread
from skimage.measure import regionprops
from skimage.exposure import rescale_intensity
from skimage.transform import rescale, resize
from tqdm import tqdm
from einops import repeat
from cell_transformations import flip_mask, rotate_image


def tifffile_to_dask(im_fp):
    imdata = zarr.open(tifffile.imread(im_fp, aszarr=True))
    if isinstance(imdata, zarr.hierarchy.Group):
        imdata = [da.from_zarr(imdata[z], chunks=(19, 10000, 10000)) for z in imdata.array_keys()]
    else:
        imdata = da.from_zarr(imdata, chunks=(19, 10000, 10000))
    return imdata


def get_channel_info():
    """Returns lists of channel names and indices that are going to be kept, as well as a dictionary mapping marker names to indices."""
    channels = [
        'DAPI',
        'AF1',
        'CD31',
        'CD45',
        'CD68',
        'Argo550',
        'CD4',
        'FOXP3',
        'CD8a',
        'CD45RO',
        'CD20',
        'PD-L1',
        'CD3e',
        'CD163',
        'E-cadherin',
        'PD-1',
        'Ki67',
        'PanCK',
        'aSMA']

    keep_channels = [ch for ch in channels if ch != "AF1" and ch != 'Argo550']
    keep_channels_idx = [i for i, ch in enumerate(channels) if ch in keep_channels]
    ch2idx = {ch: i for i, ch in enumerate(keep_channels)}

    return keep_channels, keep_channels_idx, ch2idx


def norm_if_channel(ch):
    ch = np.log(ch)
    ch[ch < 0] = 0
    ch = rescale_intensity(ch, in_range=(np.percentile(ch, 5), np.percentile(ch, 99.9)), out_range='uint8')
    return ch


def norm_if(IF):
    output = np.empty(IF.shape, dtype='uint8')
    for i, ch in tqdm(enumerate(IF)):
        output[i] = norm_if_channel(ch)
    return output


def get_mask(sid, mask_dir):
    mask_fname = os.path.join(mask_dir, f'{sid.lower()}_mesmer_cell_mask.tif')
    return imread(mask_fname)


def get_samples(data_dir, sample_name):
    '''Get unnormalized samples.'''
    print(f'loading {sample_name=}')
    if sample_name == 'CRC01':
        IF_path = os.path.join(data_dir, 'CRC01', 'registration', 'P37_S29_A24_C59kX_E15__at__20220106_014304_946511.ome.tiff')
        HE_path = os.path.join(data_dir, 'CRC01', 'registration', '18459-LSP10353-US-SCAN-OR-001 _093059-registered.ome.tif')
    else:
        sample_dir = os.path.join(data_dir, sample_name)
        IF_fname = [f for f in os.listdir(sample_dir) if f.startswith('P37') and f.endswith('-zlib.ome.tiff')][0]
        HE_fname = [f for f in os.listdir(sample_dir) if f.endswith('-registered.ome.tif')][0]
        IF_path = os.path.join(sample_dir, IF_fname)
        HE_path = os.path.join(sample_dir, HE_fname)

    IF = tifffile_to_dask(IF_path)[1]
    if sample_name == 'CRC01':
        HE = tifffile_to_dask(HE_path)[0].compute()
        assert HE.shape[0] == 3
        HE = rescale(HE, 0.5, channel_axis=0, anti_aliasing=True, preserve_range=True).astype('uint8')
    else:
        HE = tifffile_to_dask(HE_path)[1].compute()
    HE_tissue_mask = None

    return IF, HE, HE_tissue_mask


def extract_cells(IF, HE, cell_mask, sample_name, crop_size, num_channels):
    rps = regionprops(cell_mask.astype('int'))
    num_removed_from_size = 0
    num_removed_from_seg = 0
    masks, images, metadata = [], [], []
    for rp in tqdm(rps):
        min_row, min_col, max_row, max_col = rp.bbox
        if ((max_row - min_row) > crop_size) or ((max_col - min_col) > crop_size):
            num_removed_from_size += 1
            continue

        center_x, center_y = int(rp.centroid[0]), int(rp.centroid[1])
        xmin, xmax, ymin, ymax = center_x - crop_size, center_x + crop_size, center_y - crop_size, center_y + crop_size

        if_im = IF[:, xmin:xmax, ymin:ymax].copy()
        he_im = HE[:, xmin:xmax, ymin:ymax].copy()
        mask = cell_mask[xmin:xmax, ymin:ymax].copy()

        mask[mask != rp.label] = 0
        mask[mask > 0] = 1
        mask = mask.astype('uint8')
        mask = repeat(mask, 'h w -> c h w', c=num_channels)

        he_im = np.moveaxis(he_im, 0, 2)
        if_im = np.moveaxis(if_im, 0, 2)
        mask = np.moveaxis(mask, 0, 2)

        im = np.concatenate([if_im, he_im], axis=-1)

        if im[0].mean() == 0:
            num_removed_from_seg += 1
            continue

        mask = mask[:, :, 0]

        im = im[int(crop_size / 2):-int(crop_size / 2), int(crop_size / 2):-int(crop_size / 2), :]
        mask = mask[int(crop_size / 2):-int(crop_size / 2), int(crop_size / 2):-int(crop_size / 2)]

        assert im.shape == (crop_size, crop_size, num_channels), f"error im not in HxWxC, {im.shape=}"
        assert mask.shape == (crop_size, crop_size), f"error, mask shape not 32x32, {mask.shape=}"

        meta = f'{sample_name}-CellID-{rp.label}-x={center_x}-y={center_y}'
        masks.append(mask.astype('bool'))
        images.append(im.astype('uint8'))
        metadata.append(meta)

    print(f'finished processing sample {sample_name}, {num_removed_from_size=}, {num_removed_from_seg=}')
    del IF
    del cell_mask
    gc.collect()

    return masks, images, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process CRC-Orion data into single-cell HDF5 files')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to Orion CRC data directory')
    parser.add_argument('--mask-dir', type=str, required=True, help='Path to directory containing mesmer cell masks')
    parser.add_argument('--save-dir', type=str, required=True, help='Path to directory for saving output HDF5 files')
    parser.add_argument('--sample-ids', type=str, nargs='+', required=True, help='Sample IDs to process (e.g., 01 02 03)')
    parser.add_argument('--crop-size', type=int, default=32, help='Crop size for single-cell images')
    args = parser.parse_args()

    CROP_SIZE = args.crop_size
    keep_channels, keep_channels_idx, ch2idx = get_channel_info()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    for sample_id in args.sample_ids:
        sample_id = f'CRC{sample_id}'
        save_fname = f'orion_crc_dataset_sid={sample_id}.h5'
        print('retrieving samples...')
        IF, HE, HE_tissue_mask = get_samples(args.data_dir, sample_id)
        IF = IF[keep_channels_idx].compute()
        if HE.shape[-1] == 3:
            HE = np.moveaxis(HE, 2, 0)

        cell_mask = get_mask(sample_id, args.mask_dir)
        print(IF.shape, HE.shape, cell_mask.shape)

        print('normalizing IF...')
        IF = norm_if(IF)

        print('padding images...')
        pad = ((0,), (CROP_SIZE,), (CROP_SIZE,))
        IF, HE, cell_mask = np.pad(IF, pad), np.pad(HE, pad), np.pad(cell_mask, CROP_SIZE)
        print(IF.shape, HE.shape)

        print(f'extracting cells from wsi...')
        masks, images, metadata = extract_cells(IF, HE, cell_mask, sample_id, CROP_SIZE, len(keep_channels) + 3)

        print(f'saving cells to hdf5 file...')
        with h5py.File(os.path.join(args.save_dir, save_fname), 'w') as f:
            f.create_dataset('images', data=np.stack(images))
            f.create_dataset('masks', data=np.stack(masks))
            f.create_dataset('metadata', data=metadata)
