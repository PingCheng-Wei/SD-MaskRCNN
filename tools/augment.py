"""
Copyright ©2019. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Author: Mike Danielczuk

Defines data augmentation functions and provides a
method of composing them to operate upon lists of images.
"""

import argparse
import os

import numpy as np
import skimage
from autolab_core import DepthImage, YamlConfig
from tqdm import tqdm

from sd_maskrcnn.utils import mkdir_if_missing


def inject_noise(img, noise_level=0.0005):
    """
    Add a Gaussian noise to the image.
    """
    means = np.zeros(img.shape)
    std_devs = np.full(img.shape, noise_level)
    noise = np.random.normal(means, std_devs)

    # don't apply noise to some pixels
    # noise[img <= noise_threshold] = 0.0
    return img + noise


def inpaint(img):
    """
    Inpaint the image
    """
    # create DepthImage from gray version of img
    gray_img = skimage.color.rgb2gray(img)
    depth_img = DepthImage(gray_img)

    # zero out high-gradient areas and inpaint
    thresh_img = depth_img.threshold_gradients_pctile(0.95)
    inpaint_img = thresh_img.inpaint()
    return inpaint_img.data


def augment_img(img, config):
    """
    Compose augmentations.
    """
    if config["inpaint"]:
        img = inpaint(img)
    if config["inject_noise"]:
        noise_level = config["noise_level"]
        img = inject_noise(img, noise_level)
    return img


def augment(config):
    """
    Using provided image directory and output directory, perform data
    augmentation methods on each image and save the new copy to the
    output directory.
    """
    img_dir = config["img_dir"]
    out_dir = config["out_dir"]

    mkdir_if_missing(out_dir)

    print("Augmenting data in directory {}.\n".format(img_dir))
    for img_file in tqdm(os.listdir(img_dir)):
        if img_file.endswith(".png"):
            # read in image
            img_path = os.path.join(img_dir, img_file)
            img = skimage.io.imread(img_path, as_grey=True)

            # return list of augmented images and save
            new_img = augment_img(img, config)
            out_path = os.path.join(out_dir, img_file)
            skimage.io.imsave(out_path, skimage.img_as_ubyte(new_img))

    print("Augmentation complete; files saved in {}.\n".format(out_dir))


if __name__ == "__main__":

    # parse the provided configuration file and augment
    conf_parser = argparse.ArgumentParser(
        description="Augment images for SD Mask RCNN"
    )
    conf_parser.add_argument(
        "--config",
        action="store",
        default="cfg/augment.yaml",
        dest="conf_file",
        type=str,
        help="path to the configuration file",
    )
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)
    augment(config)
