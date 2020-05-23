from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from utils.detector_utils import WebcamVideoStream
import datetime
import argparse
import GUI
import sys

frame_processed = 0
score_thresh = 0.2

# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue
#app = GUI.QtWidgets.QApplication(sys.argv)
#window = GUI.Ui()
#print("before app.exec_()")
#app.exec_()
#print("after app.exec_()")

def worker_detect(input_q, output_q,classify_q, cap_params, frame_processed):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.compat.v1.Session(graph=detection_graph)
    print("> ===== in worker_detect loop, frame ", frame_processed)
    while True:
        frame = input_q.get()
        crop_img = None
        if (frame is not None):
            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)
            # draw bounding boxes
            crop_img,last_gesture = detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame)
            if crop_img is not None:
#                print("adding Crop img to classify queue")
                classify_q.put((crop_img,last_gesture))
            # add frame annotated with bounding box to queue
            output_q.put(frame)
            frame_processed += 1
        else:
            output_q.put(frame)
    sess.close()

def worker_classify(classify_q):
    print("> ===== in worker_classify loop, frame ")
    while(True):
        hand_frame = classify_q.get()
        if hand_frame[0] is not None:
            detector_utils.gesture_classifier(hand_frame[0],hand_frame[1])
   

if __name__ == '__main__':
    app = GUI.QtWidgets.QApplication(sys.argv)
    window = GUI.Ui()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type=int,
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=2,
        help='Max number of hands to detect.')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=300,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=200,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=2,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    classify_q = Queue(maxsize=args.queue_size)

    video_capture = WebcamVideoStream(
        src=args.video_source, width=args.width, height=args.height).start()

    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = args.num_hands

    print(cap_params, args)

    # spin up workers to paralleize detection.
    pool = Pool(int(args.num_workers/2), worker_detect,
                (input_q, output_q,classify_q, cap_params, frame_processed))
    pool2 = Pool(int(args.num_workers/2), worker_classify,
                (classify_q,))

    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0

#    cv2.namedWindow('Multi-Threaded Detection', cv2.WINDOW_NORMAL)

    while True:
        frame = video_capture.read()
        #cv2.waitKey(100)
        frame = cv2.flip(frame, 1)
        index += 1

        input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output_frame = output_q.get()

        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        num_frames += 1
        fps = num_frames / elapsed_time
        # print("frame ",  index, num_frames, elapsed_time, fps)

        if (output_frame is not None):
            if (args.display > 0):
                if (args.fps > 0):
                    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                     output_frame)
#                cv2.imshow('Multi-Threaded Detection', output_frame)
                window.put_frame(output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if (num_frames == 400):
                    num_frames = 0
                    start_time = datetime.datetime.now()
                else:
                    print("frames processed: ", index, "elapsed time: ",
                          elapsed_time, "fps: ", str(int(fps)))
        else:
            # print("video end")
            break
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time
    print("fps", fps)
    pool.terminate()
    pool2.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
#    app.exec_()