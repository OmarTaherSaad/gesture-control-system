# mosue-replacement-system
> all files followed with "WIP" are not ready yet or missing and will be added soon.

Graduation Project for Team 5, Computer & Systems Engineering 2020, Ain Shams University

# Description
This is a mouse replacement for traditional mouse, enabling users to control their mouse using hand gestures and hand movement

# Installation
To Start Using this system, you need to have these installed on your machine:
1. **pip** or **conda**
2. **python**

#### Steps to Install
 
1. Clone the repository.
`git clone https://github.com/OmarTaherSaad/gesture-control-system.git`
2. Install the required packages.
##### For pip users:
`pip install -r requirements.txt`
##### For conda users:
 `conda install -r requirements.txt`
 
 This will install required packages, and download all needed files. **It needs good internet connection and may take some time** (depending on your network speed).
 
# Usage

1. Open **conda** or **pip** terminal in the project path.
2. run this command for **multi-threaded** version
	`python detect_multi_threaded.py`
	
	or this command for **single threaded** version
	`python detect_multi_threaded.py --num-w 1`
	
3. Make your hand in the boundries shown in the camera preview to enable the software to detect it.

4. First hand will be used to move the mouse and will be bounded with **green rectangle**. 

5. Second hand appears into the frame will be used to do the gestures for mouse action (to click, right click .. etc.) and will be bounded with **blue rectangle**.

6. Move the first hand to move the mosue; moving to the right of screen will move the mouse to the right, same with left, top, and bottom.

# Mouse Actions
In order to do mouse actions some gestures should be made , and there pre mapped gestures to certain actions :
* Left Click 
***
![Gesture one](https://i.ibb.co/PxsWzk9/dyaa-one855original.png) 
***
* Double Click
***
![Gesture two](https://i.ibb.co/GTzHCLs/dyaa-two759original.png) 
***
* Right Click
***
![Gesture fist](https://i.ibb.co/sPXDpM0/dyaa-fist908original.png) 
***

# Development

## Hand Detector
#### We used Tensorflow Object Detection API for the detector model
### 1. Tensorflow Object Detection API
* [Installation steps](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html) Tensorflow Object Detection API

### 2. SSD Mobilenet V1
*   SSD Mobilenet V1 architecure was used : [Config file](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config)

*   [Weights](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)
### 3. Detector Model Training Notebook
* We used the follwing [Colab Notebook](https://colab.research.google.com/drive/1M5yvo2NahWi517Ha3dezRXUgAx8iQnRb?fbclid=IwAR1AeMffcjmOe0JhHLcLMFBMdBiyo-sFqsWnY4MZyTKgKleBLmcPXAJFVvM) to train our Detector Model
### 4. DataSet
We constructed our dataset by integrating [images](https://drive.google.com/drive/folders/1hQB0s_W-kOr7ZxHNQvVG-ip8RDjrfsvY?fbclid=IwAR34E9Xrl2Ap8kuHJ7RJ3p1fqnADlR3MgWJ3QiiZC3UcmjvJ0qfqNgpjT58) from 
1. Egohands Dataset 
2. Custom images we captured
### 5. Building Custom Dataset
* We capture images using trackgesture.py**(WIP)** file and save it to the local machine
* We label the captured images using [labelimg](https://github.com/tzutalin/labelImg)

## Gesture Classifier Model
#### We used keras Tensorflow for the training library and Resnet50 architecture for the classifier
### 1. Building Custom Dataset
* We used the following instructions to build the custom dataset : [Custom Dataset Building Instructions](https://drive.google.com/open?id=1mi9MiJjFMgXlPqMLLqoc-Jd1-3bt6-7H).

### 2. Preprocessing the data
*   We used hte follwing [Google Colab Notebook](https://drive.google.com/open?id=1zMfKx9vuqbESgSQxFvLHI8432eQJNh0L)**(WIP)** to preprocess the data and extract numpy arrays to use for the training process.

### 3. Training the model on the custom dataset
* We used the follwing [Google Colab Notebook](https://drive.google.com/open?id=1bMq-GQafn67xkotlfF9dReOSywc2QY1s)**(WIP)** to train our Classifier Model


