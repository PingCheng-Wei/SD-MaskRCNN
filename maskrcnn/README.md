# Mask R-CNN for Object Detection and Segmentation using TensorFlow 2.0

This [Mask-RCNN-TF2](https://github.com/PingCheng-Wei/Mask-R-CNN) project edits the original [Mask_RCNN](https://github.com/matterport/Mask_RCNN) project, which only supports TensorFlow 1.0, so that it works on TensorFlow 2.0. Based on this new project, the [Mask R-CNN](https://github.com/PingCheng-Wei/Mask-R-CNN) can be trained and tested (i.e make predictions) in TensorFlow 2.0. The Mask R-CNN model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

Compared to the source code of the old [Mask_RCNN](https://github.com/matterport/Mask_RCNN) project, the [Mask-RCNN-TF2](https://github.com/PingCheng-Wei/Mask-R-CNN) project edits the following 2 modules:

1. `model.py`
2. `utils.py`
3. `visualize.py`

The [Mask-RCNN-TF2](https://github.com/PingCheng-Wei/Mask-R-CNN) project is tested against on the following software & hardware enviroment versions:
* Ubuntu == 18.04.5 LTS 
* GPU == GeForce GTX 1080 Ti
* CUDA == 11.2
* cuDNN == 8.1 
* Python 3.7.10
  * TensorFlow 2.5.0
  * Keras 2.5.0-tf 
  * keras-nightly==2.5.0 
 
**Note that this project will not run in TensorFlow 1.0**

# Installation
1. Create a root directory (e.g. MaskRCNN) and clone this repository into it 
   ```bash
   git clone </path/to/repository>
   ```
   Inside your local root directory should look like this:
   ```
   <root directory (e.g. MaskRCNN)>/
   datasets/
   mrcnn/
     model.py
     utils.py
     ...
   samples/
     coco/
     ballon/
     ...
   setup.py
   requirements.txt
   video_demo.py
   visualize_cv2.py
   ...
   ```

2. Install Anaconda from [official link](https://docs.anaconda.com/anaconda/install/index.html)

3. Install dependencies in an enviroment in Anaconda
   * Option 1: create a new enviroment and install requirements
      ```bash
      # Please replace "myenv" with the environment name.
      conda create --name myenv
      # Install requirements
      pip3 install -r requirements.txt
      ```
   * Option 2: import directly the enviroment from [SD-MaskRCNN](https://github.com/PingCheng-Wei/SD-MaskRCNN). Download the `yml file` from this [links](https://github.com/PingCheng-Wei/SD-MaskRCNN/blob/main/environment.yml) into repository root directory, which is competible with Mask RCNN and run
      ```bash
      conda env create -f <path/to/yaml/file>
      # conda env create -f environment.yml
      ```
   * For more infomation about Anaconda enviroments managing, please check this [links](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

4. Run setup from the repository root directory
   ```bash
   python3 setup.py install
   ```
   ` Always remember to rerun this command if you have stored any change in the Mask RCNN code or enviroment !!! Otherwise the changes won't perform. `

   After running this command your repository root directory should look like this:
   ```
   <root directory (e.g. MaskRCNN)>/
   datasets/
   mrcnn/
   ...
   build/
   dist/
   mask_rcnn.egg-info/
   ...
   setup.py
   requirements.txt
   ...
   ```

5. Download `pre-trained COCO weights (mask_rcnn_coco.h5)` from the [releases page](https://github.com/matterport/Mask_RCNN/releases) or other weights you wants such as `pre-trained ballon weights (mask_rcnn_balloon.h5)` directly into the repository root directory.
   ```
   <root directory (e.g. MaskRCNN)>/
   ...
   mask_rcnn_balloon.h5
   mask_rcnn_coco.h5
   ...
   ```

6. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

   or run this to install this package with conda
   ```
   conda install -c conda-forge pycocotools
   ```

# Use the Project
Here are the steps to use the project for making predictions:

1. Download the pre-trained weights inside the root directory:
   * [releases page](https://github.com/matterport/Mask_RCNN/releases)
   * [mask_rcnn_coco.h5](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)
   * [mask_rcnn_balloon.h5](https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5)
2. Create a script for object detection and save it inside the root directory. This script is an example: [MaskRCNN/mask-rcnn-prediction.py](MaskRCNN/mask-rcnn-prediction.py).
3. Run the script.
* For more example to understand how to utilize this, you could go to the `samples folder` and explore yourself. Here is also another good explaination and implementation with balloon dataset: [Instance Segmentation with Mask R-CNN and TensorFlow with ballon dataset](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)
* To retrain your own Mask RCNN please store the dataset in `datasets folder`

The directory tree of the project is as follows:

```
<root directory (e.g. MaskRCNN)>/
assets/
datasets/
build/
dist/
mask_rcnn.egg-info/
images/
mrcnn/
  model.py
  utils.py
  ...
setup.py
requirements.txt
...
mask_rcnn_coco.h5
mask-rcnn-prediction.py
...
video_demo.py
visualize_cv2.py
...
```

The provided two scripts `video_demo.py` and `visualize_cv2.py` help you to run the Mask RCNN on a video or in real-time detection. To run the code, all you have to do is go into the repository root directory and run the following command:
```python
# Webcam real-time detection
python video_demo.py 0 

# Video detection (replace video.mp4 with the actual video file name or path/to/video.mp4)
python video_demo.py video.mp4
```

_For more understanding and explaination about maskrcnn you could read the original discription from matterport in [this link](https://github.com/matterport/Mask_RCNN)._
