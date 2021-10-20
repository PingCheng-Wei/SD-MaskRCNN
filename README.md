# Segmenting Unknown 3D Objects from Real<br/> Depth Images using Mask R-CNN Trained<br/> on Synthetic Data (SD-MaskRCNN)
This is an implementation of [SD-MaskRCNN](https://github.com/BerkeleyAutomation/sd-maskrcnn) which enables the instance segmentation for our Binpicking/DeepGrasping-Segmentation task with robot arm. The idea behind instance segmentation is to let the model achieve higher pixel-level detection instead of just traditional object detection with bonding box. This will not only benefit us with higher accuracy and precision of detection of each instance/object but also enble in our task, Binpicking, much more futher development such as center point of object detcetion, better decision of grasping point and better grasping angle. To read the original disciption, you could click [this link](https://github.com/BerkeleyAutomation/sd-maskrcnn). The final project report could be found at [here](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/Project_Report.pdf), where you could see the final testing result from the `page 29`

This repository includes:

* The Source Code of SD-MaskRCNN
* The Source Code of Mask RCNN
* Retrained weights
* Other useful scripts
* Documentations and images

This [SD-MaskRCNN](https://github.com/PingCheng-Wei/SD-MaskRCNN) project is tested against on the following software & hardware enviroment versions:

* Ubuntu == 18.04.5 LTS
* GPU == GeForce GTX 1080 Ti
* CUDA == 11.2
* cuDNN == 8.1
* Python 3.7.10
    * TensorFlow 2.5.0
    * Keras 2.5.0-tf
    * keras-nightly==2.5.0

## Table of Contents

* [Theory](https://github.com/PingCheng-Wei/SD-MaskRCNN#theory)
  * [Mask RCNN](https://github.com/PingCheng-Wei/SD-MaskRCNN#mask-rcnn)
  * [SD-MaskRCNN](https://github.com/PingCheng-Wei/SD-MaskRCNN#sd-maskrcnn)
* [Installation](https://github.com/PingCheng-Wei/SD-MaskRCNN#installation)
* [Implementation](https://github.com/PingCheng-Wei/SD-MaskRCNN#implementation)
  * [Run Detection](https://github.com/PingCheng-Wei/SD-MaskRCNN#run-detection)
  * [Train a New Model](https://github.com/PingCheng-Wei/SD-MaskRCNN#train-a-new-model)
  * [Generate a New Dataset for training](https://github.com/PingCheng-Wei/SD-MaskRCNN#generate-a-new-dataset-for-training)
  * [Generate Semantic Masks](https://github.com/PingCheng-Wei/SD-MaskRCNN#generate-semantic-masks)
  * [Rename all the image to the format "image_{index}"](https://github.com/PingCheng-Wei/SD-MaskRCNN#rename-all-the-image-to-the-format-image_index)
* [Results](https://github.com/PingCheng-Wei/SD-MaskRCNN#results)

# Theory

## Mask RCNN
In order to grasp the concept of the Mask RCNN, we have create PPT to easily explain the overall architecture and detailed functionality of each parts. Please check out or download the PPT from [this links](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/SD-MaskRCNN_Exp1.pptx) and go to "Theory of Mask RCNN" part. The experiment 1 results are also shown in the PPT. Feel free to explore yourself.

Here is an overall architecture of Mask RCNN:

![Overall architecture of Mask RCNN](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/MaskRCNN_structure/Overall%20architecture%20of%20Mask%20RCNN.png)

For more information and understanding, here are some useful links:
* [Instance Segmentation with Mask R-CNN and TensorFlow with ballon dataset](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)
* [Simple Understanding of Mask RCNN](https://alittlepain833.medium.com/simple-understanding-of-mask-rcnn-134b5b330e95)
* [Summary and Full Process of Mask RCNN Explained in Chinese](https://zhuanlan.zhihu.com/p/349584827)

## SD-MaskRCNN
Basically, the model of SD-MaskRCNN is based on mask RCNN. But with the main two following changes:

* Instead of training on RBG image, SD-MaskRCNN trains on depth images and triplicate the depth values to treat them as grayscale images
* Output classes are 2, which means background and object

Due to the ability of instance segmentation of Mask RCNN and ouput 2 classes feature, SD-MaskRCNN is capable to seperate the foreground successfully from the background and mark each different object as well as instances in same object with different labels and colors

Other contributions from SD-MaskRCNN:

* Provide a better way to set all the parameters for training & testing by inputting a config file to the model
* Provide augmentation, resize script for better training results
* Create a function to be able to run the inference on a whole testing dataset
* Create a function to filter out the BG in visualization if semantic masks are provided
* Dataset generation script, which automatically creates depth images and object masks using simulated heaps of 3D CAD models

# Installation

1. Install Nvidia Driver, CUDA and cuDNN to enable Nvidia GPU
    * Make sure what kind of TensorFlow version you need and install the corresponding version of CUDA and cuDNN. Check out [Tensorflow Build Configurations](https://www.tensorflow.org/install/source#tested_build_configurations). Also take a look at your [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) if you want.
    * Go to [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) and download the correct CUDA version you needed. Then run the downloaded file to install. `Remember to add the path variable of the bin folder to enviroment variable in your system path if you try to install on windows`
    * Go to [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive) and download the correct cuDNN version you needed. Then run the downloaded file to install. `Remember to add the path variable of the bin folder to enviroment variable in your system path if you try to install on windows`
    * Go to [Nvidia Driver Website](https://www.nvidia.com/Download/index.aspx?lang=en-us) and download the Driver corresponding to your GPU. Then run the downloaded file to install. To find out what GPU you have, run this code:
        ```bash
        lspci | grep VGA
        ```
        you will see something like this ` NVIDIA Corporation GP102 [GeForce GTX 1080 Ti] (rev a1)`
    
    * For more detailed installation process, you could visit the following links:
        * [Install CUDA and CUDNN on Windows & Linux](https://techzizou.com/install-cuda-and-cudnn-on-windows-and-linux/)
        * [CUDA, CuDNN, and Tensorflow installation on windows and Linux](https://codeperfectplus.herokuapp.com/cuda-and-tensorflow-installation-on-windows-linux)
        * [How to Install CUDA on Ubuntu 18.04](https://linoxide.com/install-cuda-ubuntu/)
        * Nvidia Official Guide:
            | |CUDA|cuDNN|
            |---|---|---|
            |Linux|[NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)|[Installing cuDNN On Linux](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-linux)|
            |Windows|[CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)|[Installing cuDNN On Windows](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)|
2. Create a root directory (e.g. SD-MaskRCNN) and clone this repository into it

   ```bash
   git clone </path/to/repository>
   ```

   or you could just download this repository and unzip the downloaded file

   Inside your local root directory should look like this:

   ```python
   # Please keep the structure like this. Otherwise it won't work !!!
   <root directory (e.g. SD-MaskRCNN)>/
   assets/
   datasets/
   cfg/
      binpicking_test.yaml
      binpicking_train.yaml
   maskrcnn/
      <the_whole_model_of_maskrcnn>
   models/
      <where_to_store_the_models_weights>
   sd_maskrcnn/
      <the_whole_model_of_sd_maskrcnn>
   tools/
      train.py
      detect.py
      ...
   environment.yml
   install.sh
   setup.py
   Dataset_generation.py
   rename_images.py
   SemanticMask_generation.py
   ...
   ```

   **Please keep the project structure like this otherwise it might not work !!!**

3. Install Anaconda from [official link](https://docs.anaconda.com/anaconda/install/index.html)

4. Install dependencies in an enviroment in Anaconda

    import directly the enviroment from the `environment.yml` file. run:

     `Please also change the "prefix" in environment.yml to your own <path/to/anaconda3/envs/sd-maskrcnn>. Currently, it is "prefix: /home/muk-pw/anaconda3/envs/sd-maskrcnn, which is my system path. Please change it to yours.`

    ```bash
    conda env create -f <path/to/environment.yml/file>
    # for example: conda env create -f environment.yml
    ```
   

    This will automatically create an enviroment named "SD-MaskRCNN", which includes all the libraries that SD-MaskRCNN needs

   * if the method above doesn't work, you then have to create a new enviroment and go into `maskrcnn folder` to install requirements. After that their might still have some libraries missing. You could only fix this problem by gradually testing and debugging to see what error message you got. It will show what library or module is missing and you then install the library with conda command in this enviroment
     ```bash
      # Please replace "myenv" with the environment name.
      conda create --name myenv
      # go into root directory of SD-MaskRCNN
      cd <path/to/root/directory>
      # go into maskrcnn
      cd maskrcnn
      # Install requirements
      pip3 install -r requirements.txt
     ```
   * For more infomation about Anaconda enviroments managing, please check this [links](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

5. Run Setup
   * Mask RCNN setup 
   ```bash
   # Enter the maskrcnn folder
   cd <path/to/maskrcnn>
   # Run the setup
   python3 setup.py install
   ```
   ` Always remember to rerun this command if you have stored any change in the Mask RCNN code or enviroment!!! Otherwise the changes won't perform. `
   * SD-MaskRCNN setup
   ```bash
   # Go back to SD-MaskRCNN folder
   cd .. 
   # or cd <path/to/SD-MaskRCNN>
   
   # intall setup
   bash install.sh
   # If dataset generation capabilities are desired, run this instead for setup
   bash install.sh generation
   ```
      This command will install the repo, and download the available pre-trained model (benchmark `sd_maskrcnn.h5` weights) to the `models` directory

6. Download the model weights and store them in the `models` directory. The followings are the available pretrained weights:
   * [releases page](https://github.com/matterport/Mask_RCNN/releases)
   * COCO pre-trained model (ResNet101): [mask_rcnn_coco.h5](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)
   * SD-MaskRCNN benchmark pre-trained model (ResNet35): [sd_maskrcnn.h5](https://berkeley.box.com/shared/static/obj0b2o589gc1odr2jwkx4qjbep11t0o.h5)
   * Binpicking Dexnet Dataset Model (ResNet101): [sd_maskrcnn_dexnet_res101.h5](https://drive.google.com/file/d/1wwdzkhvsKVHyZmIzE8V7eqVRLhQe5T7l/view?usp=sharing)
   * Binpicking Final Model (ResNet101): [sd_maskrcnn_final_res101.h5](https://drive.google.com/file/d/1F-MfKgxNb-O1H6I3bD-HEQ5ymd_ZJJrd/view?usp=sharing)

     ```
     <root directory (e.g. SD-MaskRCNN)>/
     models/
        sd_maskrcnn.h5
        mask_rcnn_coco.h5
        sd_maskrcnn_dexnet_res101.h5
        ...
     ...
     ```

7. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

   or run this to install this package with conda
   ```
   conda install -c conda-forge pycocotools
   ```

# Implementation
Make sure you are back to repository root directory of SD-MaskRCNN or run:

```
cd <path/to/SD-MaskRCNN>
```

## Run Detection
We have created a new script called `detection.py` and config file called `binpicking_test.yaml` so that SD-MaskRCNN could be able to run on single image, video, webcam (real-time) and a whole dataset either with or without ground truth instance masks. Edit the parameters in `cfg/binpicking_test.yaml` to match your need. For example, set `detect/ìmage` to 1 to enable single image detection and remember to provide the image path in `image/path` and `image/name`. For more information about the parameters, read and follow the comment in `cfg/binpicking_test.yaml`

Here are some examples to run command (GPU selection included):

```bash
python tools/detect.py
python tools/detect.py --config cfg/another.yaml
CUDA_VISIBLE_DEVICES=0 python tools/detect.py --config cfg/another.yaml
```

the default value for `--config` in `tools/detect.py` is `cfg/binpicking_test.yaml`. Feel free to change the default value to your own configuration file 

## Train a New Model
To train a new model, first [generate a new dataset for training](https://github.com/PingCheng-Wei/SD-MaskRCNN#generate-a-new-dataset-for-training). Edit `cfg/binpicking_train.yaml` so that the train path points to the dataset to train on (typically, inside this trainset root directory it should contain at least depth images folder and instance segmentation folder) and adjust training parameters such as:

* epochs
* weights, this could be "new", "last", "coco", "imagenet", "other model weights name stored in `models` folder" or custom path) 
* for your GPU (e.g., number of images per GPU, GPU count)
* dataset
    * path to trainset
    * images: folder name of depth images
    * masks: folder name of instance segmentation masks

Then, run 
```
python tools/train.py --config=cfg/binpicking_train.yaml
```

the default value for `--config` in `tools/train.py` is `cfg/train.yaml`. Feel free to change the default value to to your own configuration file 

## Generate a New Dataset for training
We have created a `Dataset_generation.py` that could automatically collect  and rename all the depth images, RGB images and instance masks from numerous folders, upon of which contains images from different object. The script will also seperate the images into training dataset or testing dataset by given the name of test object folder and create the “train_indices.npy“ and “test_indices.npy“, which includes the random indices of images

Here are some examples to run command:
```bash
# Basic
python Dataset_generation.py --root_dir=/data/Train_Set --output_root_dir=/data/Experiment1

# More setting
python Dataset_generation.py --root_dir=/data/Train_Set --output_root_dir=/data/Experiment1 --test_dataset=['Dexnet_Climbing_hold', 'Dexnet_Vase'] --read_only=Dexnet
```

This will generate the new dataset with the following struture:

```
<output root directory (e.g. /data/SD-MaskRCNN_exp1/)>/
Test/
    color_ims/
        image_000000.png
        image_000001.png
        ...
    depth_ims/
        image_000000.png
        image_000001.png
        ...
    modal_segmasks/
        image_000000.png
        image_000001.png
        ...
    segmasks_filled/
    ...
    test_indices.npy
Train/
    color_ims/
        image_000000.png
        image_000001.png
        ...
    depth_ims/
        image_000000.png
        image_000001.png
        ...
    modal_segmasks/
        image_000000.png
        image_000001.png
        ...
    segmasks_filled/
    ...
    train_indices.npy
    test_indices.npy
```

Inside the `segmasks_filled` folder is empty. We still need to run the `SemanticMask_generation.py` script to [generate semantic masks](https://github.com/PingCheng-Wei/SD-MaskRCNN#generate-semantic-masks) and store them into it if they are needed for instance in testing/detection.

For more information, please explore yourself in the script

## Generate Semantic Masks
Since SD-MaskRCNN has provided a good function to filter out the BG for better visualization if semantic masks are present, we have written a script called `SemanticMask_generation.py` to generate the semantic masks from provided root directory, which contains instance segmentation masks of a dataset and store it in folder “segmasks_filled“.

Here is an example to run the command:

```bash
python SemanticMask_generation.py --root_dir=/data/Train_Set/Dexnet_U2/Packaging/Blue_Bin/Bin_0/
```

## Rename all the image to the format "image_{index}"
To match the format requirement of SD-MaskRCNN since part of the model runs the images dependent on the index if you run the model on a dataset, we have created a script called `rename_images.py`, that automatically rename all the images in a root directory to “image_{index}“ . It also creates the indices file for testing on a dataset

Here is an example to run the command:

```bash
python rename_images.py --root_dir=/data/Train_Set/Dexnet_U2/
```

_For more understanding and explaination about SD-MaskRCNN or implementation of benchmark pre-trained model, you could read the original discription from BerkeleyAutomation in [this link](https://github.com/BerkeleyAutomation/sd-maskrcnn)._

# Results

## Dexnet Vase 
|| RGB Image | Ground Truth | Predict (no filtered) | Predict (filtered) |
|---|---|---|---|---|
| Package | ![DexnetVase_package_rgb](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/Results/DexnetVase_package_rgb.png) | ![DexnetVase_package_gt](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/Results/DexnetVase_package_gt.png) | ![DexnetVase_package_pred1](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/Results/DexnetVase_package_pred1.png) | ![DexnetVase_package_pred2](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/Results/DexnetVase_package_pred2.png) |
| Without Package | ![DexnetVase_nopackage_rgb](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/Results/DexnetVase_nopackage_rgb.png) | ![DexnetVase_nopackage_gt](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/Results/DexnetVase_nopackage_gt.png) | ![DexnetVase_nopackage_pred1](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/Results/DexnetVase_nopackage_pred1.png) | ![DexnetVase_nopackage_pred2](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/Results/DexnetVase_nopackage_pred2.png) |

## Dexnet U2
|| RGB Image | Ground Truth | Predict (no filtered) | Predict (filtered) |
|---|---|---|---|---|
| Package | ![DexnetU2_package_rgb](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/Results/DexnetU2_package_rgb.png) | ![DexnetU2_package_gt](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/Results/DexnetU2_package_gt.png) | ![DexnetU2_package_pred1](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/Results/DexnetU2_package_pred1.png) | ![DexnetU2_package_pred2](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/Results/DexnetU2_package_pred2.png) |
| Without Package | ![DexnetU2_nopackage_rgb](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/Results/DexnetU2_nopackage_rgb.png) | ![DexnetU2_nopackage_gt](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/Results/DexnetU2_nopackage_gt.png) | ![DexnetU2_nopackage_pred1](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/Results/DexnetU2_nopackage_pred1.png) | ![DexnetU2_nopackage_pred2](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/assets/Results/DexnetU2_nopackage_pred2.png) |

