# Joint Neural Networks for One-shot Object Detection (Under construction, soon to be updated)

This is the implementation of the "Joint Neural Networks for One-shot Object Recognition and Detection" thesis by Camilo Vargas.

## Requirements
* Python 3.x
* Numpy
* OpenCV
* Pytorch
* Matplotlib
* PIL
* QMUL-OpenLogo dataset (https://qmul-openlogo.github.io/)

## Usage
Set the training and testing parameters in ```python params/config.py``` file. Run the ```python main.py``` file to train/test the defined configuration.

## Results

Joint neural network performance for one-shot object detection tested on the VOC dataset, leaving four unseen classes:

Method / class | cow | sheep | cat | aeroplane | mAP
-------------|-----|-----|-----|-----|-----|
JNN | 64.7 | 51.0 | 65.2 | 43.5 | 69.1

Joint neural network performance for one-shot object detection trained on the COCO dataset, tested on the VOC dataset:

Method / class | plant | sofa | tv | car | bottle | boat | chair | person | bus | train | horse | bike | dog | bird | mbike | table | cow | sheep | cat | aero | mAP
-------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
JNN | 9.5 | 69.3 | 49.8 | 60.3 | 7.2 | 29.1 | 10.1 | 6.7 | 60.3 | 57.2 | 58.5 | 45.3 | 62.6 | 45.6 | 74.8 | 29.0 | 70.4 | 55.4 | 54.4 | 88.1 | 47.1

Testing on the top performing Open-Logodataset classes and the mAP results for the whole dataset

Class | AP / mAP
-------------|-----|
anz_text | 100.00
rbc | 98.86
blizzardentertainment | 98.34
costco | 93.26
3m | 90.61
bosch_text | 90.00
gap | 89.47
lexus | 88.92
generalelectric | 83.32
hp | 82.81
levis | 79.36
airhawk | 79.17
danone | 79.02
armitron | 77.73
google | 77.66
all | 52.84

## Reference
Joint Neural Networks for One-shot ObjectRecognition and Detection. Camilo Jose Vargas Cortes. School of Electronic Engineering and Computer Science. Queen Mary University of London. 2020.

## Examples

<img src="https://github.com/cjvargasc/JNN_detection/blob/master/imgs/3mquery.png" width="25%">-<img src="https://github.com/cjvargasc/JNN_detection/blob/master/imgs/3mtarget.png" width="25%">

<img src="https://github.com/cjvargasc/JNN_detection/blob/master/imgs/BEQuery.png" width="25%">-<img src="https://github.com/cjvargasc/JNN_detection/blob/master/imgs/BEtarget.png" width="25%">

<img src="https://github.com/cjvargasc/JNN_detection/blob/master/imgs/GEquery.04.2020.png" width="25%">-<img src="https://github.com/cjvargasc/JNN_detection/blob/master/imgs/GEtarget.png" width="25%">

<img src="https://github.com/cjvargasc/JNN_detection/blob/master/imgs/athalonquery.png" width="25%">-<img src="https://github.com/cjvargasc/JNN_detection/blob/master/imgs/athalontarget.png" width="25%">

## acknowledgement
This code is based on the following repositories:
* https://github.com/tztztztztz/yolov2.pytorch
* https://github.com/uvipen/Yolo-v2-pytorch
