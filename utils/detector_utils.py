# Utilities for object detector.
from PIL import Image
import GUI
import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict
from pynput.mouse import Button, Controller
from tensorflow.keras.models import load_model
import scipy
import tensorflow.keras.applications.vgg16 as vgg16
import tensorflow.keras.applications.vgg19 as vgg19
import tensorflow.keras.applications.resnet50 as resnet50
import math
import json
# loading model of classifier
model1 = resnet50.ResNet50(
    weights='imagenet', include_top=False, input_shape=(128, 128, 3), classes=6)
model = load_model('train91_valoss_0.6_vallacc82.hdf5')
model.load_weights('train91_valoss_0.6_vallacc82.hdf5') 

classes = ['fist', 'one', 'two', 'three', 'four', 'palm']

# taking average of last 3 frames scores for classes
pred_counter = 0
prediction_interval = 3
list_predictions = np.full((prediction_interval, len(classes)), -1.0)

# instance of mouse object
mouse = Controller()

# loading model of detector
detection_graph = tf.Graph()
sys.path.append("..")

# score threshold for showing bounding boxes.
_score_thresh = 0.8

# old_center
center_old = (0, 0)

# x_sign_old for defining last x_direction
x_sign_old = 0

# y_sign_old for defining last y_direction
y_sign_old = 0
###########################################
#parameters to be configured
###########################################
# mouse_sensitivity
sensitivity = 3
accelerator = 1
accelerator_incr = 0.2
# defining corners in screen
HOR_EXTREME = 0.4
VER_EXTREME = 0.4
# scale of screen
SCREEN_W = 1366
SCREEN_H = 768
#rectangle gui for center
top_left_x = 75
top_left_y = 42
bottom_right_x = 245
bottom_right_y = 190
#########################################
# screen_regions
hor_region_left = (0, SCREEN_W * HOR_EXTREME)
hor_region_mid = (hor_region_left[1], SCREEN_W - (SCREEN_W * HOR_EXTREME))
hor_region_right = (hor_region_mid[1], SCREEN_W)
ver_region_top = (0, SCREEN_H * VER_EXTREME)
ver_region_mid = (ver_region_top[1], SCREEN_H - (SCREEN_H * VER_EXTREME))
ver_region_bottom = (ver_region_mid[1], SCREEN_H)

# last_gesture
last_gesture = ""
#flag of scrolling
flag_scroll = 0
#last number of hands detected
last_num_hands = 0
current_num_hands = 0
#last region detected
LEFT_REGION = 0
RIGHT_REGION = 1
TOP_REGION = 2
BOTTOM_REGION = 3
CENTER_REGION = 4
last_region = CENTER_REGION
current_region = CENTER_REGION

MODEL_NAME = 'trained-inference-graphs'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'object-detection.pbtxt')

NUM_CLASSES = 1
# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess



def change_settings():
    global sensitivity, accelerator_incr, HOR_EXTREME, VER_EXTREME
    global SCREEN_W, SCREEN_H ,top_left_x ,top_left_y, bottom_right_x, bottom_right_y 
    global hor_region_left, hor_region_mid, hor_region_right, ver_region_top, ver_region_mid, ver_region_bottom

    with open('config.json') as file:
        data = json.load(file)
        sensitivity = data["sensitivity"]
        accelerator_incr = data["acceleration"]
        HOR_EXTREME = data["Horizontal_Margin"]
        VER_EXTREME = data["Vertical_Margin"]
        SCREEN_W = data["resolution_in_x"]
        SCREEN_H = data["resolution_in_y"]
    top_left_x = int(320 * HOR_EXTREME)
    top_left_y = int(240 * VER_EXTREME)
    bottom_right_x = int(320 - 320 * HOR_EXTREME )
    bottom_right_y = int(240 -240 * VER_EXTREME)
    hor_region_left = (0, SCREEN_W * HOR_EXTREME)
    hor_region_mid = (hor_region_left[1], SCREEN_W - (SCREEN_W * HOR_EXTREME))
    hor_region_right = (hor_region_mid[1], SCREEN_W)
    ver_region_top = (0, SCREEN_H * VER_EXTREME)
    ver_region_mid = (ver_region_top[1], SCREEN_H - (SCREEN_H * VER_EXTREME))
    ver_region_bottom = (ver_region_mid[1], SCREEN_H)

# draw the detected bounding boxes on the images
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    global center_old, left_hand_flag, right_hand_flag, last_num_hands, current_num_hands, counter

    change_settings()
        
    last_gesture = ""
    # initializing centers with max numbers
    new_centers = [(2000, 2000), (2000, 2000)]
    #drawing rectangle to show center where the mouse stops
    img_to_show = image_np.copy()
    cv2.rectangle(image_np, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), (255, 0, 0), 3, 1)
    current_num_hands = 0
    # detecting hands
    # looping through hands detected
    for i in range(num_hands_detect):
        if (scores[i] > 0.6):  # Score for how likely is it really a hand
            current_num_hands += 1
            # Get hand boundries
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            # crop image and resize it and pass it to the model
            center = (((int(((right-left)/2) + left))/im_width)*SCREEN_W,
                      ((int(((bottom-top)/2)+top))/im_height)*SCREEN_H)
            cv2.circle(image_np, (((int(((right-left)/2) + left))),((int(((bottom-top)/2)+top)))), radius=5, color=(0, 0, 255), thickness=-1)
            new_centers[i] = center
            
#    print("current num hands",current_num_hands)
#    print("last num hands before condition",last_num_hands)
    if current_num_hands == 2 and last_num_hands == 1:
        last_gesture = "nothing"
        last_num_hands = 2
#        print("last gesture in palm condition",last_gesture)
    elif current_num_hands == 1:
        last_num_hands = 1
#    print("last num hands after condition",last_num_hands)

    # determining difference between old center and new cnters of both hands
    distance_diff = [0, 0]
    mouse_index = 0
    distance_diff[0] = math.sqrt(
        (new_centers[0][0]-center_old[0])**2+(new_centers[0][1]-center_old[1])**2)
    distance_diff[1] = math.sqrt(
        (new_centers[1][0]-center_old[0])**2+(new_centers[1][1]-center_old[1])**2)
    if distance_diff[0] < distance_diff[1]:
        mouse_index = 0
    else:
        mouse_index = 1

    # the smallest difference is the tracking hand
    if scores[mouse_index] > 0.6:
        (left, right, top, bottom) = (boxes[mouse_index][1] * im_width, boxes[mouse_index][3] * im_width,
                                      boxes[mouse_index][0] * im_height, boxes[mouse_index][2] * im_height)
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))
        cv2.rectangle(image_np, p1, p2, (0, 255, 0), 3, 1)
        # mouse tracking function calling
        mouse_control_option2(new_centers[mouse_index])
        center_old = new_centers[mouse_index]


    # the largest difference is the classifing hand
    if scores[(mouse_index+1) % 2] > 0.6:
        (left, right, top, bottom) = (boxes[(mouse_index+1) % 2][1] * im_width, boxes[(mouse_index+1) % 2][3] * im_width,
                                      boxes[(mouse_index+1) % 2][0] * im_height, boxes[(mouse_index+1) % 2][2] * im_height)
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))
        cv2.rectangle(image_np, p1, p2, (0, 0, 255), 3, 1)
        crop_img = crop_hands (left, right, top, bottom, img_to_show)
        return crop_img, last_gesture
        # gesture classifier function calling
#        gesture_classifier(left, right, top, bottom, img_to_show)
    return None,""
# Mouse tracking function
# Option 1: Move mouse with hand movement
def mouse_control_option1(center, center_old, left, right, top, bottom, image_np):
    global x_sign_old, y_sign_old, sensitivity_x, sensitivity_y, accelerator
    x_sign, y_sign = 0, 0
    # difference between center of current frame and last frame
    difference_x = int(center[0]-center_old[0])
    difference_y = int(center[1]-center_old[1])
    # threshold for movement
    threshold = 30

    if abs(difference_x) > threshold:  # Should move in x axis
        x_sign = 1 if difference_x > threshold else -1
    else:
        x_sign = 0

    if abs(difference_y) > threshold:  # Should move in y axis
        y_sign = 1 if int(difference_y) > threshold else -1
    else:
        y_sign = 0
    # increase sensitivity of mouse if it is moving in the same direction
    if (x_sign == x_sign_old or y_sign == y_sign_old) and (x_sign != 0 or y_sign != 0):
        sensitivity_x += 2*accelerator
        sensitivity_y += 4*accelerator
        accelerator += 1
    else:
        sensitivity_x, sensitivity_y = 3, 5
        accelerator = 1

    mouse.move(x_sign*sensitivity_x, y_sign*sensitivity_y)
    x_sign_old, y_sign_old = x_sign, y_sign

# Mouse tracking function
# Option 2: Increment mouse position with hand movement:
#When hand goes to right: mouse moves to right .. etc.
def mouse_control_option2(center):
    global sensitivity, accelerator, accelerator_incr, hor_region_left, last_region, current_region
    global hor_region_mid, hor_region_right, ver_region_top, ver_region_mid, ver_region_bottom
    # if cv2.waitKey(1) & 0xFF == ord('+'):
    #     print("current accelerator increment: ",accelerator_incr)
    #     accelerator_incr = float(input("Enter desired accelerator increment:"))

    center_x = center[0]
    center_y = center[1]
    if center_x < hor_region_left[1]:
        mouse.move(-1*sensitivity*accelerator, 0*sensitivity*accelerator)
        accelerator += accelerator_incr
        current_region = LEFT_REGION
    elif center_x > hor_region_right[0]:
        mouse.move(1*sensitivity*accelerator, 0*sensitivity*accelerator)
        accelerator += accelerator_incr
        current_region = RIGHT_REGION
    elif center_y < ver_region_top[1]:
        mouse.move(0*sensitivity*accelerator, -1*sensitivity*accelerator)
        accelerator += accelerator_incr
        current_region = TOP_REGION
    elif center_y > ver_region_bottom[0]:
        mouse.move(0*sensitivity*accelerator, 1*sensitivity*accelerator)
        accelerator += accelerator_incr
        current_region = BOTTOM_REGION
    else:
        mouse.move(0*sensitivity, 0*sensitivity)
        current_region = CENTER_REGION
    if current_region != last_region:
        accelerator = 1
    last_region = current_region


def crop_hands (left, right, top, bottom, image_np):
    crop_img = image_np[int(top):int(bottom), int(left):int(right)]
    if(not crop_img.any()):
        return None
    return crop_img
# gesture classifier function using the model for prediction
def gesture_classifier(crop_img,l_gesture):
    global pred_counter, list_predictions, counter, last_gesture, flag_scroll
    
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    crop_img = Image.fromarray(crop_img)
    crop_img = crop_img.resize((128, 128), Image.ANTIALIAS)
    crop_img = np.asarray(crop_img)
    crop_img = crop_img.astype('float32')
    crop_img = np.expand_dims(crop_img, axis=0)
    crop_img = resnet50.preprocess_input(crop_img)
    crop_img_pred = model1.predict(crop_img)
    # new predictions
    y_new = model.predict(crop_img_pred)
    list_predictions[pred_counter] = y_new[0]
    pred_counter = (pred_counter+1) % prediction_interval
    # get the class with the most number of votes in the last three frames
    gesture, y = predict(list_predictions)
#    print("last gesture before conditions: ",last_gesture)
#    print(y_new)
    if l_gesture != "":
        last_gesture = l_gesture
    if gesture > 0.5:
        print(classes[y])
        if classes[y] == "one" and last_gesture != "one":
            mouse.click(Button.left)
            last_gesture = "one"
        elif classes[y] == "three" and last_gesture != "three":
            mouse.click(Button.right)
            last_gesture = "three"
        elif classes[y] == "two" and last_gesture != "two":
            mouse.click(Button.left, 2)
            last_gesture = "two"
        elif classes[y] == "palm" :
            mouse.scroll(0, 80)
            last_gesture = "palm"
        elif classes[y] == "fist":
            mouse.scroll(0, -80)
            last_gesture = "fist"  
        

# Calculating average of votes of last three frames to get the most accurate class

def predict(prediction):
    sum_pred = 0
    out = []
    max_temp = 0
    max_index = 0
    for i in range(len(classes)):
        sum_pred = 0
        for j in range(prediction_interval):
            sum_pred += prediction[j][i]
        avg = sum_pred/prediction_interval
        out.append(avg)
        if avg > max_temp:
            max_temp = avg
            max_index = i
    return max_temp, max_index


# Show fps value on image.
def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


# Code to thread reading camera input.
# Source : Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def size(self):
        # return size of the capture device
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
