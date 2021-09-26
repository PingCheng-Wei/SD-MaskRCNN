"""
Renaming all of the images in the directory in "image_{index(:06)}"
so that they could then be used for training and testing for SD-MasKRCNN
Also automatically generate the indices numpy file, which contains the
indices for either training or testing
-------------------------------------------------------------------------
run from the commend line as such:
    # by giving the root directory of your datasets
    in windows: python rename:images.py --root_dir=/path/to/dataset
    in Linux:   python3 rename:images.py --root_dir=/path/to/dataset
"""

import os
import argparse
import numpy as np


def rename_images(ROOT_DIR):
    
    # make sure ROOT_DIR is absolute path
    if not os.path.isabs(ROOT_DIR):
        ROOT_DIR = os.path.abspath(ROOT_DIR)
    
    # iterate through all the files and directories
    for root, dirs, files in os.walk(ROOT_DIR):
        
        # sort the files to make sure the series are same and correct upon all type of images !!!
        files.sort()
        
        # get into the current directory
        os.chdir(root)
        # initialize the number count of images
        total_images = 0

        # start rename process
        for i in range(len(files)):

            # seperate the root and the ext of the file
            file_root, file_ext = os.path.splitext(files[i])

            # search for .png and .jpg file
            if file_ext == '.png' or file_ext == '.jpg':

                # add amount of images
                total_images += 1

                # rename the file root
                file_root = 'image_{:06}'.format(i)
                # get the new full file name
                new_file_name = '{}{}'.format(file_root, file_ext)
                try:
                    # rename the file
                    os.rename(files[i], new_file_name)
                except:
                    print('Can\'t rename the file "{}" in {} to "{}" because "{}" exists already.'
                          .format(files[i], root, new_file_name, new_file_name))
                    print('ignore it and continue...\n')
                    continue

        print('renamed {} images in "{}" directory'.format(total_images, root))

        # create the numpy indices file and save it
        if total_images != 0:
            root_head, root_tail = os.path.split(root)
            os.chdir(root_head)
            np.save('{}_indices'.format(root_tail), np.arange(total_images))
            print('stored the created numpy indices file in "{}" directory.'.format(root_head))

    # go back to the root directory
    os.chdir(ROOT_DIR)
    print('\nfinished renaming all images')
    print('We are now back to "{}" directory'.format(ROOT_DIR))


if __name__ == '__main__':
    # Parse comment line arguments
    parser = argparse.ArgumentParser(description='Renaming all images for SD-MaskRCNN')
    parser.add_argument('--root_dir', required=True,
                        metavar="/path/to/dataset",
                        help='root directory of the dataset')
    args = parser.parse_args()
    print('ROOT_DIR: {}'.format(args.root_dir))

    # start renaming
    rename_images(args.root_dir)
    print('done!')



