from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap,QImage
import cv2
import sys
from PIL import Image
import json
  
resolution_in_x = 1920
resolution_in_y = 1080
Horizontal_Margin = 0.4 
Vertical_Margin = 0.4 
sensitivity = 3
acceleration = 0.2
closing_flag = 0

class Ui(QtWidgets.QDialog):
    def __init__(self):
        global resolution_in_x, resolution_in_y, Horizontal_Margin, Vertical_Margin, sensitivity, acceleration
        with open('config.json') as file:
            data = json.load(file)
            sensitivity = data["sensitivity"]
            acceleration = data["acceleration"]
            Horizontal_Margin = data["Horizontal_Margin"]
            Vertical_Margin = data["Vertical_Margin"]
            resolution_in_x = data["resolution_in_x"]
            resolution_in_y = data["resolution_in_y"]
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('gui.ui', self)  # Load the .ui file
        #Reset
        self.reset_button=self.findChild(QtWidgets.QPushButton,"reset_button")
        self.reset_button.clicked.connect(self.Reset)
        #screen_resolution
        self.screen_resolution_combo=self.findChild(QtWidgets.QComboBox,"screen_combo")
        self.screen_resolution_combo.view().pressed.connect(self.handleTheResolution)
        
        #Horizontal Margin
        self.Horizontal_Margin_slider=self.findChild(QtWidgets.QSlider,"horizontal_slider")
        self.Horizontal_Margin_slider.valueChanged.connect(self.handleHorizontalMargin)
        self.Horizontal_Margin_slider.setValue(Horizontal_Margin/0.05)
        #Vertical Margin
        self.Vertical_Margin_slider=self.findChild(QtWidgets.QSlider,"vertical_slider")
        self.Vertical_Margin_slider.valueChanged.connect(self.handleVerticalMargin)
        self.Vertical_Margin_slider.setValue(Vertical_Margin/0.05)

        #sensitivity_slider
        self.sensitivity_slider=self.findChild(QtWidgets.QSlider,"sensitivity_slider")
        self.sensitivity_slider.valueChanged.connect(self.handleSensitivity)
        self.sensitivity_slider.setValue(sensitivity)
		#Acceleration_slider
        self.acceleration_slider=self.findChild(QtWidgets.QSlider,"acceleration_slider")
        self.acceleration_slider.valueChanged.connect(self.handleAcceleration)
        self.acceleration_slider.setValue(acceleration/0.1)

        #Apply_button
        self.apply_button=self.findChild(QtWidgets.QPushButton,"apply_button")
        self.apply_button.clicked.connect(self.Apply)
        self.show() # Show the GUI
    def Reset(self):
        #Reset to default configuration
        global resolution_in_x, resolution_in_y, Horizontal_Margin, Vertical_Margin, sensitivity, acceleration
        print("RESET")
        resolution_in_x = 1366
        resolution_in_y = 768
        Horizontal_Margin = 0.4 
        Vertical_Margin = 0.4 
        sensitivity = 3
        acceleration = 0.2
        self.Horizontal_Margin_slider.setValue(Horizontal_Margin/0.05)
        self.Vertical_Margin_slider.setValue(Vertical_Margin/0.05)
        self.sensitivity_slider.setValue(sensitivity)
        self.acceleration_slider.setValue(acceleration/0.1)
        paramters = {
             "resolution_in_x" : resolution_in_x,
             "resolution_in_y": resolution_in_y,
             "Horizontal_Margin" : Horizontal_Margin,
             "Vertical_Margin" : Vertical_Margin,
             "sensitivity" : sensitivity ,
             "acceleration" : acceleration
             }
        with open("config.json","w") as output :
            json.dump(paramters,output)

    def handleTheResolution(self,index):
        global resolution_in_x, resolution_in_y
        print("RESOULTION NOW IS ")
        item=self.screen_resolution_combo.model().itemFromIndex(index)
        resolution_in_x,resolution_in_y=item.text().split("x")
        resolution_in_x=resolution_in_x[:len(resolution_in_x)-1]
        resolution_in_y=resolution_in_y[1:]
        resolution_in_x=int(resolution_in_x)
        resolution_in_y=int(resolution_in_y)
        print(resolution_in_x)
        print(resolution_in_y)
        print(item.text())
    def handleHorizontalMargin(self):
        global Horizontal_Margin
        print("HORIZONTALMARGIN IS")
        print(self.Horizontal_Margin_slider.value())
        Horizontal_Margin = self.Horizontal_Margin_slider.value() * 0.05
    def handleVerticalMargin(self):	
        global Vertical_Margin
        print("VERTICALMARGIN")
        print(self.Vertical_Margin_slider.value())
        Vertical_Margin = self.Vertical_Margin_slider.value() * 0.05
    def handleSensitivity(self):
        global sensitivity
        print("SENSITIVITY IS")
        print(self.sensitivity_slider.value())
        sensitivity = self.sensitivity_slider.value()
    def handleAcceleration(self):
        global acceleration
        print("ACCELERATION IS ")
        print(self.acceleration_slider.value())
        acceleration =  self.acceleration_slider.value() * 0.1
    def Apply(self):
        print("Apply")
        paramters = {
                     "resolution_in_x" : resolution_in_x,
                     "resolution_in_y": resolution_in_y,
                     "Horizontal_Margin" : Horizontal_Margin,
                     "Vertical_Margin" : Vertical_Margin,
                     "sensitivity" : sensitivity ,
                     "acceleration" : acceleration
                     }
        with open("config.json","w") as output :
            json.dump(paramters,output)

    def put_frame(self,img):	
        img = Image.fromarray(img, 'RGB')
        img = img.convert("RGBA")
        data = img.tobytes("raw","RGBA")
        qim = QImage(data, img.size[0], img.size[1], QImage.Format_ARGB32)
        pix = QPixmap.fromImage(qim)
        self.video_stream.setPixmap(pix)
        
    def closeEvent(self, event):
        global closing_flag
        closing_flag = 1
        event.accept()
	

