# Installation Instruction --- Adapted from Agile3D
Adaptation by Emily Steiner, Codey Sun, Liyuan Zhu
Reachable at easteine@stanford.edu, 

### Step 0: setup Files
Clone Agile3D:
```shell
git clone https://github.com/ywyue/AGILE3D.git
cd AGILE3D
```

Unzip the project files provided. Within the Agile3D repo, replace the `interactive_tool` directory with the unzipped version (replacing all files). The files are as follows: 
code_root/
└── interactive_tool/
    └── dataloader.py: inherited from AGILE3D to lead scenes
    └── gui.py: inherited from AGILE3D original 2D gui
    └── Interactive_segmentation_user.py: main interaction class to load model and receive input from GUI or server (modified to default to server)
    └── server.py: websocket server which replaces 2D GUI to communicate colors, clicks and scene information
    └── test_client.py: test client used to test the websocket server without VR client 
    └── utils.py: inherited utils from Agile3D 
└── Unity/


### Step 1: create an environment
```shell
conda create -n agile3d python=3.10
conda activate agile3d
```
### Step 2: install pytorch
```shell
# adjust your cuda version and corresponding torch and torchvision version accordingly!
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```
### Step 3: install Minkowski
We tested using CUDA 12.6 with torch 2.6 using a modified version of [MinkowskiEngine for CUDA 12](https://github.com/GradientSpaces/MinkowskiEngine). If using CUDA 11 or CPU version, please refer to the original AGILE3D installation instructions [here](https://github.com/ywyue/AGILE3D/blob/main/installation.md)

3.1 Prepare:
```shell
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade numpy
pip install torch ninja
sudo apt install build-essential python3-dev libopenblas-dev
```
3.2 Install:
```shell
# adjust your cuda path!
export CUDA_HOME=/usr/local/cuda

git clone git@github.com:GradientSpaces/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install
```

### Step 4: install other packages
```shell
pip install open3d
pip install websockets
```

## Preparing the Backend Model  

### Step 1: download pretained model
Download the [**model**](https://polybox.ethz.ch/index.php/s/RnB1o8X7g1jL0lM) and move it to the ```weights``` folder.

The model was only trained on [ScanNet40](http://www.scan-net.org/) training set, but it can also be used to segment scenes from other datasets, e.g., [S3DIS](http://buildingparser.stanford.edu/dataset.html), [ARKitScenes](https://github.com/apple/ARKitScenes) and even outdoor scans, [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/).

### Step 2: download sample data
[**Sample data link**](https://polybox.ethz.ch/index.php/s/HMhuyJwJkPXxP3f)

The data should be organized as follows:
```
code_root/
└── data/
    └── interactive_dataset/
        ├── scene_*****/
        |    ├── scan.ply
        |    └── label.ply (optional)
        ├── scene_*****/
        ├── ...
        └── scene_*****/

```
Note:
- ```scan.ply```: the 3D scan file, which can be a mesh or a point cloud file.
- ```label.ply``` (optional): the label file which should contain a 'label' attribute that indicates the instance id (starting from 1, 2, 3 ...) of each point. This file is optional. If provided, the system will automatically record the segmentation IoU.

Here we provide some samples from ScanNet, S3DIS, KITTI-360 (with label) and samples from ARKitScenes (without label). **You may also want to try your own scans!** Note for large-scale scans, you may need to crop them into a smaller part in case out of memory.


## Running the Application 
### Step 1: Run the Backend Model and Server
Run  the following command:
```shell
python run_UI.py --user_name=test_user --pretraining_weights=weights/checkpoint1099.pth --dataset_scenes=data/interactive_dataset
```
### Step 1.1: Check the Server with the test client 
```shell
python interactive_tool/test_client.py 
```
