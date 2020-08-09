# Joint Neural Networks for One-shot Object Detection

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

Joint neural network performance for one-shot object detection using the OpenLogo dataset with a confidence threshold of 0.5.

Method | TPR | FPR | acc | Pr | AUC | mAP(%)
-------------|-----|-----|-----|-----|-----|-----|
Proposed | 0.261 | 0.001 | 0.997 | 0.094 | 0.624 | 22.64

top 5 mAP detection results for the Open-Logo dataset classes

Class | mAP(%) 
-------------|-----|
skechers | 72.50
bosch_text | 48.57
blizzardentertainment | 46.46
costco | 43.54
3m | 40.72

## Reference
Joint Neural Networks for One-shot ObjectRecognition and Detection. Camilo Jose Vargas Cortes. School of Electronic Engineering and Computer Science. Queen Mary University of London. 2020.

## Examples

<img src="https://github.com/cjvargasc/JNN_detection/blob/master/imgs/3mquery.png" width="25%">-<img src="https://github.com/cjvargasc/JNN_detection/blob/master/imgs/3mtarget.png" width="25%">

<img src="https://github.com/cjvargasc/JNN_detection/blob/master/imgs/BEQuery.png" width="25%">-<img src="https://github.com/cjvargasc/JNN_detection/blob/master/imgs/BEtarget.png" width="25%">

<img src="https://github.com/cjvargasc/JNN_detection/blob/master/imgs/GEquery.04.2020.png" width="25%">-<img src="https://github.com/cjvargasc/JNN_detection/blob/master/imgs/GEtarget.png" width="25%">

<img src="https://github.com/cjvargasc/JNN_detection/blob/master/imgs/athalonquery.png" width="25%">-<img src="https://github.com/cjvargasc/JNN_detection/blob/master/imgs/athalontarget.png" width="25%">
