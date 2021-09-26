'''
copy all the images from a given root directory to a given output
directory with a new name "image_{index(:06)}" so that they could
then be used for training and testing for SD-MasKRCNN.
Also automatically generate the indices numpy file, which contains the
indices for either training or testing
-------------------------------------------------------------------------
run from the commend line as such:
    # by giving the root directory of your datasets
    in windows: python: Dataset_generation.py --root_dir=/path/to/dataset ----output_root_dir=/path/to/store
    in Linux:   python3: Dataset_generation.py --root_dir=/path/to/dataset ----output_root_dir=/path/to/store

    example:
        python Dataset_generation.py --root_dir=/data/Train_Set --output_root_dir=/data/Experiment1
'''

import os
import sys
import argparse

import glob
import shutil
import numpy as np
from sklearn.model_selection import train_test_split


def copy_rename_images(path, output_dir, images_count):
    '''
    iterate through all folder to find the image and copy
    as well as rename it into the ouput directory
    # Arguments
        path: absolute path, root directory of data
        output_dir: absolute path, where to store the copy of images
        images_count: list, list of number of each type of images
    '''

    # iterate through all the matched files in a given path
    for root, dirs, files in os.walk(path):

        # check the folder that we need
        root_head, root_tail = os.path.split(root)

        # ignore the Train folder and all subfolders
        if root.endswith(os.path.join('Train', 'DepthMaps')) or \
                root.endswith(os.path.join('Train', 'SemanticMasks')):
            continue
        
        # sort the files to make sure the series are same and correct upon all type of images !!!
        files.sort()

        # check file in current root directory
        for file in files:
            # seperate the root and the ext of the file
            file_root, file_ext = os.path.splitext(file)

            # search for .png and .jpg file
            if file_ext == '.png' or file_ext == '.jpg':

                # get the corresponding output path
                # add up image count and rename the file
                if root_tail == 'DepthMaps':
                    output_path = os.path.join(output_dir, 'depth_ims')
                    images_count[0] += 1  # total_depth
                    # get the new full file name, image index start from 0
                    file_root = 'image_{:06}'.format(images_count[0] - 1)
                    new_file_name = '{}{}'.format(file_root, file_ext)

                elif root_tail == 'DepthMaps_800_800':
                    output_path = os.path.join(output_dir, 'depth_ims_800_800')
                    images_count[1] += 1  # total_depth_800
                    # get the new full file name, image index start from 0
                    file_root = 'image_{:06}'.format(images_count[1] - 1)
                    new_file_name = '{}{}'.format(file_root, file_ext)

                elif root_tail == 'RGBMasks':
                    output_path = os.path.join(output_dir, 'color_ims')
                    images_count[2] += 1  # total_rgb
                    # get the new full file name, image index start from 0
                    file_root = 'image_{:06}'.format(images_count[2] - 1)
                    new_file_name = '{}{}'.format(file_root, file_ext)

                elif root_tail == 'RGBMasks_800_800':
                    output_path = os.path.join(output_dir, 'color_ims_800_800')
                    images_count[3] += 1  # total_rgb_800
                    # get the new full file name, image index start from 0
                    file_root = 'image_{:06}'.format(images_count[3] - 1)
                    new_file_name = '{}{}'.format(file_root, file_ext)

                elif root_tail == 'SegmentationMasks':
                    output_path = os.path.join(output_dir, 'modal_segmasks')
                    images_count[4] += 1  # total_segmask
                    # get the new full file name, image index start from 0
                    file_root = 'image_{:06}'.format(images_count[4] - 1)
                    new_file_name = '{}{}'.format(file_root, file_ext)

                elif root_tail == 'SegmentationMasks_800_800':
                    output_path = os.path.join(output_dir, 'modal_segmasks_800_800')
                    images_count[5] += 1  # total_segmask_800
                    # get the new full file name, image index start from 0
                    file_root = 'image_{:06}'.format(images_count[5] - 1)
                    new_file_name = '{}{}'.format(file_root, file_ext)

                else:
                    # don't need images from other folders
                    continue

                # get the absolute path of the old file and new file for the copy function later
                old_file_abspath = os.path.join(root, file)
                new_file_abspath = os.path.join(output_path, new_file_name)

                try:
                    # copy the image to output directory with the new file name
                    shutil.copy(old_file_abspath, new_file_abspath)

                except:
                    print('Can\'t rename the file "{}" in {} to "{}" because "{}" exists already in {}.'
                          .format(file, root, new_file_name, new_file_name, output_path))
                    print('ignore it and continue...\n')
                    continue
        
    return images_count


def dataset_generation(root_dir, output_root_dir, test_dataset, read_only):
    '''
    automatically create the whole dataset for the experiment
    # Arguments
        root_dir: relative path and absolute path, where you read the data for the generation
        output_root_dir: relative path and absolute path, where to store the generated dataset
        read_only: string regex ['all', 'Dexnet', 'Part'], specify only from which folder names of all data to read
        test_dataset: full name of folder, which would be consider for test dataset generation
    '''

    assert read_only in ['all', 'Dexnet', 'Part'], "Argument --read_only could only be ['all', 'Dexnet', 'Part']"

    # make sure root_dir is absolute path
    if not os.path.isabs(root_dir):
        root_dir = os.path.abspath(root_dir)

    # make sure output directory is absolute and exists
    if not os.path.isabs(output_root_dir):
        output_root_dir = os.path.abspath(output_root_dir)
    if not os.path.exists(output_root_dir):
        os.mkdir(output_root_dir)

    # create all the subfolders that we need in output directory
    subfolders = ['Train', 'Test']
    subsubfolders = ['color_ims', 'depth_ims', 'modal_segmasks', 'segmasks_filled',
                     'color_ims_800_800', 'depth_ims_800_800', 'modal_segmasks_800_800',
                     'segmasks_filled_800_800']
    # start creating subpath
    for subfolder in subfolders:
        subpath = os.path.join(output_root_dir, subfolder)
        if not os.path.exists(subpath):
            os.mkdir(subpath)
        # start creating subsubpath
        for subsubfolder in subsubfolders:
            subsubpath = os.path.join(subpath, subsubfolder)
            if not os.path.exists(subsubpath):
                os.mkdir(subsubpath)

    # initialize each total images for test
    total_test_depth = 0
    total_test_depth_800 = 0
    total_test_rgb = 0
    total_test_rgb_800 = 0
    total_test_segmask = 0
    total_test_segmask_800 = 0
    totol_test_images = [total_test_depth, total_test_depth_800, total_test_rgb,
                         total_test_rgb_800, total_test_segmask, total_test_segmask_800]

    # initialize each total images for training
    total_train_depth = 0
    total_train_depth_800 = 0
    total_train_rgb = 0
    total_train_rgb_800 = 0
    total_train_segmask = 0
    total_train_segmask_800 = 0
    totol_train_images = [total_train_depth, total_train_depth_800, total_train_rgb,
                          total_train_rgb_800, total_train_segmask, total_train_segmask_800]

    for fname in os.listdir(root_dir):
        # get the file or directory path
        path = os.path.join(root_dir, fname)

        # read only the data with specific name
        if os.path.isdir(path) and fname.startswith(read_only):
            # generate test dataset
            if fname in test_dataset:
                # copy to test output folder
                output_path = os.path.join(output_root_dir, 'Test')
                print('generating test dataset from {}...'.format(fname))
                totol_test_images = copy_rename_images(path, output_path, totol_test_images)
                print('finished copying and renaming {} images of each type from {}'.format(totol_test_images, fname))

            # generate training dataset
            else:
                # copy to train output folder
                output_path = os.path.join(output_root_dir, 'Train')
                print('generating training dataset from {}...'.format(fname))
                totol_train_images = copy_rename_images(path, output_path, totol_train_images)
                print('finished copying and renaming {} images of each type from {}'.format(totol_train_images, fname))

        # read through all the data
        elif os.path.isdir(path) and read_only == 'all':
            # generate test dataset
            if fname in test_dataset:
                # copy to test output folder
                output_path = os.path.join(output_root_dir, 'Test')
                print('generating test dataset from {}...'.format(fname))
                totol_test_images = copy_rename_images(path, output_path, totol_test_images)
                print('finished copying and renaming {} images of each type from {}'.format(totol_test_images, fname))

            # generate training dataset
            else:
                # copy to train output folder
                output_path = os.path.join(output_root_dir, 'Train')
                print('generating training dataset from {}...'.format(fname))
                totol_train_images = copy_rename_images(path, output_path, totol_train_images)
                print('finished copying and renaming {} images of each type from {}'.format(totol_train_images, fname))

    # TODO
    # generate train and test indices numpy file for training
    if totol_train_images[0] != 0:
        # randomly split the indices
        train, test = train_test_split(np.arange(totol_train_images[0]), test_size=0.2, random_state=42)

        idx_dir = os.path.join(output_root_dir, 'Train')
        os.chdir(idx_dir)
        # store the indices file
        np.save('train_indices', np.sort(train))
        np.save('test_indices', np.sort(test))
        print('\nstored the created training indices file in "{}" directory.'.format(idx_dir))

    # generate test indices numpy file for testing
    if totol_test_images[0] != 0:
        idx_dir = os.path.join(output_root_dir, 'Test')
        os.chdir(idx_dir)
        # store the indices file
        np.save('test_indices', np.arange(totol_test_images[0]))
        print('stored the created testing indices file in "{}" directory.'.format(idx_dir))

    # go back to the root directory
    os.chdir(root_dir)
    print('\nfinished copying and renaming all images')
    print('We are now back to "{}" directory'.format(os.getcwd()))


if __name__ == '__main__':
    # Parse comment line arguments
    parser = argparse.ArgumentParser(
        description='Copying and Renaming all images for SD-MaskRCNN')
    parser.add_argument('--root_dir', required=True,
                        metavar="/path/to/dataset",
                        help='root directory of the dataset')
    parser.add_argument('--output_root_dir', required=True,
                        metavar="/path/to/output",
                        help='root directory to store the dataset')
    parser.add_argument('--test_dataset', required=False,
                        default=['Dexnet_Climbing_hold', 'Part_PW2', 'Part_PW12b'],
                        metavar="list of folder name for testing",
                        help='data folder that is considered as testing data')
    parser.add_argument('--read_only', required=False,
                        default='all',
                        metavar="could only be ['all', 'Dexnet', 'Part']",
                        help='to read only the data folder with given regex')
    args = parser.parse_args()
    print('ROOT_DIR: {}'.format(args.root_dir))

    # start renaming
    dataset_generation(args.root_dir, args.output_root_dir,
                       args.test_dataset, args.read_only)
    print('done!')

