'''
Given a root directory of dataset which contains the SegmentationMask folder,
this script will automatically generate the semantic segmentation masks for each image
'''

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import os
import sys
import argparse


def semanticmask_generation(root_dir):
    # make sure root_dir is absolute path
    if not os.path.isabs(root_dir):
        root_dir = os.path.abspath(root_dir)
    
    segmask_path = os.path.join(root_dir, 'SegmentationMasks')
    segmask_800_path = os.path.join(root_dir, 'SegmentationMasks_800_800')
    semanticmask_path = os.path.join(root_dir, 'segmasks_filled')
    semanticmask_800_path = os.path.join(root_dir, 'segmasks_filled_800_800')
    
#     segmask_path = os.path.join(root_dir, 'modal_segmasks')
#     segmask_800_path = os.path.join(root_dir, 'modal_segmasks_800_800')
#     semanticmask_path = os.path.join(root_dir, 'segmasks_filled')
#     semanticmask_800_path = os.path.join(root_dir, 'segmasks_filled_800_800')

    # make sure path exists
    if not os.path.exists(semanticmask_path):
        os.mkdir(semanticmask_path)
    if not os.path.exists(semanticmask_800_path):
        os.mkdir(semanticmask_800_path)

    # iterate through the files in folder
    for file in os.listdir(segmask_path):
        # separate the root and the ext of the file
        file_root, file_ext = os.path.splitext(file)
        # search for .png and .jpg file
        if file_ext == '.png' or file_ext == '.jpg':

            # read the image
            img_path = os.path.join(segmask_path, file)
            img = skimage.io.imread(img_path)

            # add this condition to make sure that the input mask is 2D
            if img.shape[-1] == 4 and img.ndim == 3:
                img = img[..., :3]  # convert 4-channel to 3-channel
            # convert 3D image to 2D image, because we must need 2D Mask for SD-MaskRCNN
            if img.shape[-1] == 3:
                img = img[:, :, 0]

            # find the BG, Box, Package label
            img_BG = img == 0
            img_Box = img == 13
            img_Package = img == 21
            # create the filter to filter out BG, Box, Package
            img_filter = img_BG + img_Box + img_Package

            # filter instance(1) and no instance(0)
            SemanticMask = np.where(img_filter, 0, 1)

            # store the image in corresponding directory
            os.chdir(semanticmask_path)
            skimage.io.imsave(file, SemanticMask)

    # iterate through the files in folder
    for file in os.listdir(segmask_800_path):
        # seperate the root and the ext of the file
        file_root, file_ext = os.path.splitext(file)
        # search for .png and .jpg file
        if file_ext == '.png' or file_ext == '.jpg':

            # read the image
            img_path = os.path.join(segmask_800_path, file)
            img = skimage.io.imread(img_path)

            # add this condition to make sure that the input mask is 2D
            if img.shape[-1] == 4 and img.ndim == 3:
                img = img[..., :3]  # convert 4-channel to 3-channel
            # convert 3D image to 2D image, because we must need 2D Mask for SD-MaskRCNN
            if img.shape[-1] == 3:
                img = img[:, :, 0]

            # find the BG, Box, Package label
            img_BG = img == 0
            img_Box = img == 13
            img_Package = img == 21
            # create the filter to filter out BG, Box, Package
            img_filter = img_BG + img_Box + img_Package

            # filter instance(1) and no instance(0)
            SemanticMask = np.where(img_filter, 0, 1)

            # store the image in corresponding directory
            os.chdir(semanticmask_800_path)
            skimage.io.imsave(file, SemanticMask)

if __name__ == '__main__':
    # Parse comment line arguments
    parser = argparse.ArgumentParser(
        description='Copying and Renaming all images for SD-MaskRCNN')
    parser.add_argument('--root_dir', required=True,
                        metavar="/path/to/dataset",
                        help='root directory of the dataset which contains instance masks folder')
    args = parser.parse_args()
    print('ROOT_DIR: {}'.format(args.root_dir))

    print('Start generating the semantic masks ...')
    semanticmask_generation(args.root_dir)
    print('done!')