from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap,QImage,QFont,QPalette,QBrush
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

class Ui(QtWidgets.QWidget):
    def __init__(self,size):
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
        self.setWindowTitle("Gesture Mouse")
        self.gen_size = int(size.height() / 24)
    
        #Prepare Font
        main_font = QFont("Arial", int(self.gen_size/5), QFont.Bold) 

        #Webcam Screem
        v_layout = QtWidgets.QVBoxLayout() # mainWindow layout
        pix = QPixmap("gui_images/logo.jpg")
        pix = pix.scaledToWidth(320)
        pix = pix.scaledToHeight(240)
        self.video_stream = QtWidgets.QLabel()
        self.video_stream.setPixmap(pix)
        v_layout.addWidget(self.video_stream)

        #Resolution Row
        h_layout_res = QtWidgets.QHBoxLayout() # Placeholder for a row
        res_pix = QPixmap("gui_images/communications.png")
        res_pix = res_pix.scaledToWidth(self.gen_size)
        res_pix = res_pix.scaledToHeight(self.gen_size)
        self.res_icon = QtWidgets.QLabel()
        self.res_icon.setPixmap(res_pix)
        self.res_label = QtWidgets.QLabel("Resolution")
        self.res_label.setFont(main_font)
        self.screen_combo = QtWidgets.QComboBox()
        self.screen_combo.addItem("1366 * 768")
        self.screen_combo.addItem("1920 * 1080")
        self.screen_combo.view().pressed.connect(self.handleTheResolution)

        h_layout_res.addWidget(self.res_icon)
        h_layout_res.addWidget(self.res_label)
        h_layout_res.addItem(QtWidgets.QSpacerItem(10, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))
        v_layout.addLayout(h_layout_res)
        v_layout.addWidget(self.screen_combo)


        #Sensitivity Row
        h_layout_sens = QtWidgets.QHBoxLayout() # Placeholder for a row
        sens_pix = QPixmap("gui_images/arrows.png")
        sens_pix = sens_pix.scaledToWidth(self.gen_size)
        sens_pix = sens_pix.scaledToHeight(self.gen_size)
        self.sens_icon = QtWidgets.QLabel()
        self.sens_icon.setPixmap(sens_pix)
        self.sens_label = QtWidgets.QLabel("Sensitivity")
        self.sens_label.setFont(main_font)
        self.sens_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.sens_slider.setTickInterval(1)
        self.sens_slider.setMinimum(1)
        self.sens_slider.setMaximum(10)
        self.sens_slider.valueChanged.connect(self.handleSensitivity)
        self.sens_slider.setValue(sensitivity)

        h_layout_sens.addWidget(self.sens_icon)
        h_layout_sens.addWidget(self.sens_label)
        h_layout_sens.addItem(QtWidgets.QSpacerItem(10, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))
        v_layout.addLayout(h_layout_sens)
        v_layout.addWidget(self.sens_slider)

        #Acceleration Row
        h_layout_acc = QtWidgets.QHBoxLayout() # Placeholder for a row
        acc_pix = QPixmap("gui_images/arrows.png")
        acc_pix= acc_pix.scaledToWidth(self.gen_size)
        acc_pix= acc_pix.scaledToHeight(self.gen_size)
        self.acc_icon = QtWidgets.QLabel()
        self.acc_icon.setPixmap(acc_pix)
        self.acc_label = QtWidgets.QLabel("Acceleration")
        self.acc_label.setFont(main_font)
        self.acc_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.acc_slider.setTickInterval(1)
        self.acc_slider.setMinimum(1)
        self.acc_slider.setMaximum(10)
        self.acc_slider.valueChanged.connect(self.handleAcceleration)
        self.acc_slider.setValue(acceleration/0.1)

        h_layout_acc.addWidget(self.acc_icon)
        h_layout_acc.addWidget(self.acc_label)
        h_layout_acc.addItem(QtWidgets.QSpacerItem(10, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))
        v_layout.addLayout(h_layout_acc)
        v_layout.addWidget(self.acc_slider)

        #Horizontal Margin
        h_layout_horz_margin = QtWidgets.QHBoxLayout() # Placeholder for a row
        horz_margin_pix = QPixmap("gui_images/horizontal.png")
        horz_margin_pix= horz_margin_pix.scaledToWidth(self.gen_size)
        horz_margin_pix= horz_margin_pix.scaledToHeight(self.gen_size)
        self.horz_margin_icon = QtWidgets.QLabel()
        self.horz_margin_icon.setPixmap(horz_margin_pix)
        self.horz_margin_label = QtWidgets.QLabel("Horizontal Margin")
        self.horz_margin_label.setFont(main_font)
        self.horz_margin_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.horz_margin_slider.setTickInterval(1)
        self.horz_margin_slider.setMinimum(1)
        self.horz_margin_slider.setMaximum(8)
        self.horz_margin_slider.valueChanged.connect(self.handleHorizontalMargin)
        self.horz_margin_slider.setValue(Horizontal_Margin/0.05)
        
        h_layout_horz_margin.addWidget(self.horz_margin_icon)
        h_layout_horz_margin.addWidget(self.horz_margin_label)
        h_layout_horz_margin.addItem(QtWidgets.QSpacerItem(10, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))
        v_layout.addLayout(h_layout_horz_margin)
        v_layout.addWidget(self.horz_margin_slider)

        #Vertical Margin
        h_layout_vert_margin = QtWidgets.QHBoxLayout() # Placeholder for a row
        vert_margin_pix = QPixmap("gui_images/vertical.png")
        vert_margin_pix = vert_margin_pix.scaledToWidth(self.gen_size)
        vert_margin_pix = vert_margin_pix.scaledToHeight(self.gen_size)
        self.vert_margin_icon = QtWidgets.QLabel()
        self.vert_margin_icon.setPixmap(vert_margin_pix)
        self.vert_margin_label = QtWidgets.QLabel("Vertical Margin")
        self.vert_margin_label.setFont(main_font)
        self.vert_margin_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.vert_margin_slider.setTickInterval(1)
        self.vert_margin_slider.setMinimum(1)
        self.vert_margin_slider.setMaximum(8)
        self.vert_margin_slider.valueChanged.connect(self.handleVerticalMargin)
        self.vert_margin_slider.setValue(Vertical_Margin/0.05)

        h_layout_vert_margin.addWidget(self.vert_margin_icon)
        h_layout_vert_margin.addWidget(self.vert_margin_label)
        h_layout_vert_margin.addItem(QtWidgets.QSpacerItem(10, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))
        v_layout.addLayout(h_layout_vert_margin)
        v_layout.addWidget(self.vert_margin_slider)

        h_layout_btns = QtWidgets.QHBoxLayout() # Placeholder for a row
        #Apply button
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.clicked.connect(self.Apply)

        #Reset Button
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.Reset)

        h_layout_btns.addWidget(self.apply_button)
        h_layout_btns.addWidget(self.reset_button)
        v_layout.addLayout(h_layout_btns)
        
        self.setLayout(v_layout)

        #Set Background
        oImage = QImage("gui_images/background.jpg")
        sImage = oImage.scaled(self.sizeHint())                   # resize Image to widgets size
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(sImage))                        
        self.setPalette(palette)

        self.setFixedSize(self.sizeHint()) #Disable resize
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
        self.horz_margin_slider.setValue(Horizontal_Margin/0.05)
        self.vert_margin_slider.setValue(Vertical_Margin/0.05)
        self.sens_slider.setValue(sensitivity)
        self.acc_slider.setValue(acceleration/0.1)
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
        item=self.screen_combo.model().itemFromIndex(index)
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
        print(self.horz_margin_slider.value())
        Horizontal_Margin = self.horz_margin_slider.value() * 0.05

    def handleVerticalMargin(self):	
        global Vertical_Margin
        print("VERTICALMARGIN")
        print(self.vert_margin_slider.value())
        Vertical_Margin = self.vert_margin_slider.value() * 0.05

    def handleSensitivity(self):
        global sensitivity
        print("SENSITIVITY IS")
        print(self.sens_slider.value())
        sensitivity = self.sens_slider.value()

    def handleAcceleration(self):
        global acceleration
        print("ACCELERATION IS ")
        print(self.acc_slider.value())
        acceleration =  self.acc_slider.value() * 0.1

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
	

