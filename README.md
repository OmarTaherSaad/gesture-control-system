# gesture-control-system
 Graduation Project for Team 5, Computer & Systems Engineering 2020, Ain Shams University

#Description
This is a mouse replacement for traditional mouse, enabling users to control their mouse using hand gestures.

# Installation
To Start Using this system, you need to have these installed on your machine:
1- **pip** or **conda**
2- **python**

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

1.  Open **conda** or **pip** terminal in the project path.
2. run this command for single-thread version
	`python detect_single_threaded.py`
	
	or this command for multi-thread version
	`python detect_multi_threaded.py`
	
3.  Make your hand in the boundries shown in the camera preview to enable the software to detect it.

4. First hand will be used to move mouse, second hand appears into detector will be used to do the gestures (to click, right click .. etc.)

5. Move the first hand to move the mosue; moving to the right of screen will move the mouse to the right, same with left, top, and bottom.
