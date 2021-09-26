"""
Test Usage Notes:

Please edit "cfg/binpicking_test.yaml" to specify the necessary parameters for that task.

Run this file with the tag --config [config file name]
if different config from the default location (cfg/binpicking_test.yaml).

Here is an example run command (GPU selection included):
    - python tools/detect.py
    - python tools/detect.py --config cfg/another.yaml
    - CUDA_VISIBLE_DEVICES=0 python tools/detect.py --config cfg/another.yaml
"""

import argparse
import os
import sys
import datetime
from copy import copy
import skimage.io
import skimage.color

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from autolab_core import YamlConfig
from tensorflow.compat.v1.keras.backend import set_session
from mrcnn import model as modellib
from mrcnn import visualize
from tqdm import tqdm

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
print(ROOT_DIR)
# Import SD Mask-RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from sd_maskrcnn import utils
from sd_maskrcnn.coco_benchmark import coco_benchmark
from sd_maskrcnn.dataset import ImageDataset
from sd_maskrcnn.model import SDMaskRCNNModel
from sd_maskrcnn.supplement_benchmark import s_benchmark

from mrcnn import utils as utilslib


def detect(config):

    print('Testing the model.')

    # Create new directory for outputs
    output_dir = config["output_dir"]
    utils.mkdir_if_missing(output_dir)

    # Save config in output directory
    config.save(os.path.join(output_dir, config["save_conf_name"]))

    # Create the model (weight-loading in constructor)
    model = SDMaskRCNNModel("inference", config["model"])

    ############ decide which data type to detect ############

    if config["detect"]["image"]:
        img_fullpath = os.path.join(config["image"]["path"], config["image"]["name"])
        print("Running on {}".format(config["image"]["name"]))

        # Read image
        img = skimage.io.imread(img_fullpath)
        # make sure image is 3D and with 3 channels
        channel_count = config["model"]["settings"]["image_channel_count"]
        if channel_count < 4 and img.shape[-1] == 4 and img.ndim == 3:
            img = img[..., :3]
        elif channel_count == 3 and img.ndim == 3 and img.shape[-1] == 1:
            img = skimage.color.gray2rgb(img)
        elif channel_count == 3 and img.ndim == 2:
            img = skimage.color.gray2rgb(img)

        # Read bin_mask
        bin_mask = None
        if config["image"]["bin_mask_path"] is not None:
            bin_mask = skimage.io.imread(config["image"]["bin_mask_path"])
            # add this condition to make sure that the input mask is 2D
            if bin_mask.shape[-1] == 4 and bin_mask.ndim == 3:
                bin_mask = bin_mask[..., :3]  # convert 4-channel to 3-channel
            # convert 3D image to 2D image, because we must need 2D Mask for SD-MaskRCNN
            # we don't need semantic mask with repeated 3 channel
            if bin_mask.shape[-1] == 3:
                bin_mask = bin_mask[:, :, 0]
            # convert back 2D to 3D to match the image dimension
            bin_mask = bin_mask[..., np.newaxis]

        # detect the instance
        masks_detect, mask_info = model.detect(img, bin_mask=bin_mask,
                                               overlap_thresh=config["image"]["overlap_thresh"])
        # Must transpose from (n, h, w) to (h, w, n)
        if masks_detect.ndim == 3:
            masks = np.transpose(masks_detect, (1, 2, 0))
        else:
            masks = masks_detect
        # resize the image to match the masks size
        img, _, _, _, _ = utilslib.resize_image(
            img,
            min_dim=model.mask_config.IMAGE_MIN_DIM,
            min_scale=model.mask_config.IMAGE_MIN_SCALE,
            max_dim=model.mask_config.IMAGE_MAX_DIM,
            mode=model.mask_config.IMAGE_RESIZE_MODE,
        )

        print("VISUALIZING PREDICTIONS")

        # Visualize
        scores = mask_info["scores"] if config["image"]["show_scores_pred"] else None
        show_bbox = config["image"]["show_bbox_pred"]
        show_class = config["image"]["show_class_pred"]
        fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        fig.add_axes(ax)
        visualize.display_instances(
            img,
            mask_info["rois"],
            masks,
            mask_info["class_ids"],
            ["bg", "obj"],
            ax=ax,
            scores=scores,
            show_bbox=show_bbox,
            show_class=show_class,
        )
                                            # or config["image"]["name"]
        file_name = os.path.join(output_dir, "pred_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now()))
        fig.savefig(file_name, transparent=True, dpi=300)
        plt.close()
        
        print("done!")
        print("the result of the image is store in {}".format(file_name))


    if config["detect"]["video"]:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(config["video"]["path"])
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = os.path.join(output_dir, "sd-maskrcnn_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now()))
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))
        """
        Define a function to convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
            - fig: a matplotlib figure
        return a numpy 3D array of RGBA values
        """
        def fig2np(fig):
            # draw the renderer
            fig.canvas.draw()
            # Get the RGBA buffer from the figure
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            buf.shape = (w, h, 4)
            # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
            buf = np.roll(buf, 3, axis=2)
            return buf

        # start detecting
        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                masks_detect, mask_info = model.detect(image, bin_mask=None)
                
                # Must transpose from (n, h, w) to (h, w, n)
                if masks_detect.ndim == 3:
                    masks = np.transpose(masks_detect, (1, 2, 0))
                else:
                    masks = masks_detect
                # resize the image to match the masks size
                image, _, _, _, _ = utilslib.resize_image(
                    image,
                    min_dim=model.mask_config.IMAGE_MIN_DIM,
                    min_scale=model.mask_config.IMAGE_MIN_SCALE,
                    max_dim=model.mask_config.IMAGE_MAX_DIM,
                    mode=model.mask_config.IMAGE_RESIZE_MODE,
                )

                # Visualize
                scores = mask_info["scores"] if config["video"]["show_scores_pred"] else None
                show_bbox = config["video"]["show_bbox_pred"]
                show_class = config["video"]["show_class_pred"]
                fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
                ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
                fig.add_axes(ax)
                visualize.display_instances(
                    image,
                    mask_info["rois"],
                    masks,
                    mask_info["class_ids"],
                    ["bg", "obj"],
                    ax=ax,
                    scores=scores,
                    show_bbox=show_bbox,
                    show_class=show_class,
                )
                # convert figure back to image with numpy array data type
                masked_image = fig2np(fig=fig)
                # RGB -> BGR to save image to video
                masked_image = masked_image[..., ::-1]

                # run real time detection
                if config["video"]["path"] == 0:
                    cv2.imshow("Instance Segmentation with SD-MaskRCNN", masked_image)
                    # click "q" to quit the running
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                # run video detection
                else:
                    # Add image to video writer
                    vwriter.write(masked_image)
                    count += 1

        if config["video"]["path"] == 0:
            vcapture.release()
            cv2.destroyWindow("masked_image")

        else:
            vwriter.release()
            print("Saved to ", file_name)


    if config["detect"]["dataset"]:

        # Create dataset
        test_dataset = ImageDataset(config)
        test_dataset.load(config["dataset"]["indices"])
        test_dataset.prepare()

        vis_config = copy(config)
        # vis_config["dataset"]["images"] = "depth_ims"
        # vis_config["dataset"]["masks"] = "modal_segmasks"
        vis_dataset = ImageDataset(config)
        vis_dataset.load(config["dataset"]["indices"])
        vis_dataset.prepare()

        # Overarching benchmark function just creates the directory
        # code that actually produces outputs should be plug-and-play
        # depending on what kind of benchmark function we run.

        # If we want to remove bin pixels, pass in the directory with
        # those masks.
        if config["mask"]["remove_bin_pixels"]:
            bin_mask_dir = os.path.join(
                config["dataset"]["path"], config["mask"]["bin_masks"]
            )
            overlap_thresh = config["mask"]["overlap_thresh"]
        else:
            bin_mask_dir = None
            overlap_thresh = 0

        # Create predictions and record where everything gets stored.
        pred_mask_dir, pred_info_dir, gt_mask_dir = model.detect_dataset(
            config["output_dir"],
            test_dataset,
            bin_mask_dir,
            overlap_thresh,
        )

        '''
        Depends on whether the ground truth for detection dataset is provided or not,
        we could use
            - Part A => "visualize_inferences" for not provided
        or 
            - Part B =>"visualize_predictions"
            - Part B =>"config["vis"]["ground_truth"]" for provided
        '''
        # # Part A
        # # show and store the inference of instance segmentation results
        # if config["vis"]["predictions"]:
        #     visualize_inferences(
        #         config["output_dir"],
        #         vis_dataset,
        #         model.mask_config,
        #         pred_mask_dir,
        #         pred_info_dir,
        #         show_bbox=config["vis"]["show_bbox_pred"],
        #         show_scores=config["vis"]["show_scores_pred"],
        #         show_class=config["vis"]["show_class_pred"],
        #     )
        # print("Saved detection output of dataset to {}.\n".format(config["output_dir"]))
        ################################################################################

        # Part B
        # show and store the segmentation results
        if config["vis"]["predictions"]:
            visualize_predictions(
                config["output_dir"],
                vis_dataset,
                model.mask_config,
                pred_mask_dir,
                pred_info_dir,
                show_bbox=config["vis"]["show_bbox_pred"],
                show_scores=config["vis"]["show_scores_pred"],
                show_class=config["vis"]["show_class_pred"],
            )
        if config["vis"]["ground_truth"]:
            visualize_gts(
                config["output_dir"],
                vis_dataset,
                model.mask_config,
                show_scores=False,
                show_bbox=config["vis"]["show_bbox_gt"],
                show_class=config["vis"]["show_class_gt"],
            )
        print("Saved detection output of dataset to {}.\n".format(config["output_dir"]))
        ##################################################################################

        if config["vis"]["s_bench"]:
            s_benchmark(
                config["output_dir"],
                vis_dataset,
                model.mask_config,
                pred_mask_dir,
                pred_info_dir,
            )

        # # compute the AP and AR using the COCO API in comparison with gt
        # ap, ar = coco_benchmark(pred_mask_dir, pred_info_dir, gt_mask_dir)
        # return ap, ar



# add extra function for the case without gt of testdataset
# either use 'visualize_inferences' or 'visualize_predictions' functions
def visualize_inferences(
    run_dir,
    dataset,
    inference_config,
    pred_mask_dir,
    pred_info_dir,
    show_bbox=True,
    show_scores=True,
    show_class=True,
):
    """Visualizes predictions."""
    # Create subdirectory for prediction visualizations
    vis_dir = os.path.join(run_dir, "vis")
    utils.mkdir_if_missing(vis_dir)

    # Feed images into model one by one. For each image visualize predictions
    image_ids = dataset.image_ids

    print("VISUALIZING INFERENCES")

    for image_id in tqdm(image_ids):

        ###################################################################################
        # Load the image by using dataset.load_image() to read
        # Resize the image with utilslib.resize_image() to match the prediciton mask size
        image = dataset.load_image(image_id)
        image, _, _, _, _ = utilslib.resize_image(
            image,
            min_dim=inference_config.IMAGE_MIN_DIM,
            min_scale=inference_config.IMAGE_MIN_SCALE,
            max_dim=inference_config.IMAGE_MAX_DIM,
            mode=inference_config.IMAGE_RESIZE_MODE,
        )
        ###################################################################################

        if inference_config.IMAGE_CHANNEL_COUNT == 1:
            image = np.repeat(image, 3, axis=2)
        elif inference_config.IMAGE_CHANNEL_COUNT > 3:
            image = image[:, :, :3]

        # load mask and info
        r = np.load(
            os.path.join(pred_info_dir, "image_{:06}.npy".format(image_id)),
            allow_pickle=True,
        ).item()
        r_masks = np.load(
            os.path.join(pred_mask_dir, "image_{:06}.npy".format(image_id))
        )
        # Must transpose from (n, h, w) to (h, w, n)
        if r_masks.ndim == 3:
            r["masks"] = np.transpose(r_masks, (1, 2, 0))
        else:
            r["masks"] = r_masks
        # Visualize
        scores = r["scores"] if show_scores else None
        fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        fig.add_axes(ax)
        visualize.display_instances(
            image,
            r["rois"],
            r["masks"],
            r["class_ids"],
            ["bg", "obj"],
            ax=ax,
            scores=scores,
            show_bbox=show_bbox,
            show_class=show_class,
        )
        file_name = os.path.join(vis_dir, "vis_{:06d}".format(image_id))
        fig.savefig(file_name, transparent=True, dpi=300)
        plt.close()


def visualize_predictions(
    run_dir,
    dataset,
    inference_config,
    pred_mask_dir,
    pred_info_dir,
    show_bbox=True,
    show_scores=True,
    show_class=True,
):
    """Visualizes predictions."""
    # Create subdirectory for prediction visualizations
    vis_dir = os.path.join(run_dir, "vis")
    utils.mkdir_if_missing(vis_dir)

    # Feed images into model one by one. For each image visualize predictions
    image_ids = dataset.image_ids

    print("VISUALIZING PREDICTIONS")

    for image_id in tqdm(image_ids):

        # Load image and ground truth data and resize for net
        image, _, _, _, _ = modellib.load_image_gt(
            dataset, inference_config, image_id, use_mini_mask=False
        )

        if inference_config.IMAGE_CHANNEL_COUNT == 1:
            image = np.repeat(image, 3, axis=2)
        elif inference_config.IMAGE_CHANNEL_COUNT > 3:
            image = image[:, :, :3]

        # load mask and info
        r = np.load(
            os.path.join(pred_info_dir, "image_{:06}.npy".format(image_id)),
            allow_pickle=True,
        ).item()
        r_masks = np.load(
            os.path.join(pred_mask_dir, "image_{:06}.npy".format(image_id))
        )
        # Must transpose from (n, h, w) to (h, w, n)
        if r_masks.ndim == 3:
            r["masks"] = np.transpose(r_masks, (1, 2, 0))
        else:
            r["masks"] = r_masks
        # Visualize
        scores = r["scores"] if show_scores else None
        fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        fig.add_axes(ax)
        visualize.display_instances(
            image,
            r["rois"],
            r["masks"],
            r["class_ids"],
            ["bg", "obj"],
            ax=ax,
            scores=scores,
            show_bbox=show_bbox,
            show_class=show_class,
        )
        file_name = os.path.join(vis_dir, "vis_{:06d}".format(image_id))
        fig.savefig(file_name, transparent=True, dpi=300)
        plt.close()


def visualize_gts(
    run_dir,
    dataset,
    inference_config,
    show_bbox=True,
    show_scores=False,
    show_class=True,
):
    """Visualizes gts."""
    # Create subdirectory for gt visualizations
    vis_dir = os.path.join(run_dir, "gt_vis")
    utils.mkdir_if_missing(vis_dir)

    # Feed images one by one
    image_ids = dataset.image_ids

    print("VISUALIZING GROUND TRUTHS")
    for image_id in tqdm(image_ids):
        # Load image and ground truth data and resize for net
        image, _, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset, inference_config, image_id, use_mini_mask=False
        )

        if inference_config.IMAGE_CHANNEL_COUNT == 1:
            image = np.repeat(image, 3, axis=2)

        # Visualize
        scores = np.ones(gt_class_id.size) if show_scores else None
        fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        fig.add_axes(ax)
        visualize.display_instances(
            image,
            gt_bbox,
            gt_mask,
            gt_class_id,
            ["bg", "obj"],
            scores,
            ax=ax,
            show_bbox=show_bbox,
            show_class=show_class,
        )
        file_name = os.path.join(vis_dir, "gt_vis_{:06d}".format(image_id))
        fig.savefig(file_name, transparent=True, dpi=300)
        plt.close()


if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and benchmark
    conf_parser = argparse.ArgumentParser(
        description="Benchmark SD Mask RCNN model"
    )
    conf_parser.add_argument(
        "--config",
        action="store",
        default="cfg/binpicking_test.yaml",
        dest="conf_file",
        type=str,
        help="path to the configuration file",
    )
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)

    # Set up tf session to use what GPU mem it needs and benchmark
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=tf_config) as sess:
        set_session(sess)
        detect(config)
