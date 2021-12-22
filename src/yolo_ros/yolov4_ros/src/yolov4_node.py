#!/usr/bin/python3
import sys
import os
import platform
import getpass
from unicodedata import name
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import yolo_detect
import yolo_detect_trt
from image_obj_msgs.msg import ImageObj
from image_obj_msgs.msg import ImageBox
import time
import threading
import queue

def main():
    rospy.init_node('yolo_detector')
    #========================================= ROS params =========================================
    sub_image_topic = rospy.get_param("~input_image_topic", default="/d400/color/image_raw")
    cfgfile = rospy.get_param("~model_file", default="/home/autoware/shared_dir/workspace/yolo_ws/src/yolov4_ros/models/yolov4.cfg")
    trtfile = rospy.get_param("~trt_file", default="/home/autoware/shared_dir/workspace/yolo_ws/src/yolov4_ros/models/yolov4.trt")
    weightfile = rospy.get_param("~prototxt_file", default="/home/autoware/shared_dir/workspace/yolo_ws/src/yolov4_ros/models/yolov4.weights")
    namesfile = rospy.get_param("~namesfile", default="/home/autoware/shared_dir/workspace/yolo_ws/src/yolov4_ros/data/coco.names")
    use_cuda = rospy.get_param("~use_cuda", default=True)
    gui = False
    # For every log_interval seconds we will print performance stats for each topic
    log_interval = rospy.get_param('~log_interval', 3.0)
    #==============================================================================================
    #====================================== More ROS params =======================================
    # Model path and hyperparameters are provided by the specifed net by default, but can be changed
    # e.g., one can tell the desired (maximal) numbers of keypoints by setting keypoint_number,
    #       or filtering out low-quality keypoints by setting a high value of keypoint_threshold
    #==============================================================================================
    # net = FeatureNet(net_path,'mobilenet_v2')
    net =  yolo_detect.Detector(cfgfile,weightfile,namesfile,use_cuda)
    # net =  yolo_detect_trt.Detector_trt(trtfile, namesfile, (320, 320))
    node = Node(net, gui, log_interval)
    # for topic in topics.split(','):
    node.subscribe(sub_image_topic)
    rospy.spin()

class Node():
    def __init__(self, net, gui, log_interval):
        self.net = net
        self.gui = gui
        self.log_interval = log_interval
        self.cv_bridge = CvBridge()
        self.feature_publishers = {}
        self.subscribers = {}
        self.latest_msgs = {}
        self.latest_msgs_lock = threading.Lock()
        self.stats = {}
        self.stats_lock = threading.Lock()
        self.result_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.worker)
        self.worker_thread.start()
        self.publisher_thread = threading.Thread(target=self.publisher)
        self.publisher_thread.start()

    def subscribe(self, topic):
        # callback = lambda msg: self.callback(msg, topic)
        # rospy.Subscriber('/cam0/image_raw', Image, callback)
        base_topic = '/'.join(topic.split('/')[:-1])
        self.feature_publishers[topic] = rospy.Publisher( '/image/detected', ImageObj, queue_size=1)
        print(base_topic + '/features')
        self.stats[topic] = {'received': 0, 'processed': 0, 'last_time': None}
        with self.latest_msgs_lock:
            self.latest_msgs[topic] = None
        callback = lambda msg: self.callback(msg, topic)
        self.subscribers[topic] = rospy.Subscriber(topic, Image, callback, queue_size=1, buff_size=2**24)

    def callback(self, msg, topic):
        # keep only the lastest message
        with self.latest_msgs_lock:
            self.latest_msgs[topic] = msg
        with self.stats_lock:
            self.stats[topic]['received'] += 1
        # print(self.stats[topic]['received'])

    def worker(self):
        while not rospy.is_shutdown():
            no_new_msg = True
            # take turn to process each topic
            for topic in self.latest_msgs.keys():
                with self.latest_msgs_lock:
                    msg = self.latest_msgs[topic]
                    self.latest_msgs[topic] = None
                if msg is not None:
                    self.process(msg, topic)
                    with self.stats_lock:
                        self.stats[topic]['processed'] += 1
                    no_new_msg = False
                self.print_stats(topic)
            if no_new_msg: time.sleep(0.01)

    def publisher(self):
        while not rospy.is_shutdown():
            if self.result_queue.qsize() > 5:
                rospy.logwarn_throttle(1, 'WOW! Inference is faster than publishing' +
                    ' (%d unpublished result in the queue)\n' % self.result_queue.qsize() +
                    'Please add more publisher threads!')
            try:
                res = self.result_queue.get(timeout=.5)
            except queue.Empty:
                continue
            features = res['features']
            topic = res['topic']
            image = res['image']
            header = res['header']
            self.result_queue.task_done()
            feature_msg = features_to_ros_msg(features, header)
            self.feature_publishers[topic].publish(feature_msg)

    def print_stats(self, topic):
        now = rospy.Time.now()
        if self.stats[topic]['last_time'] is None:
            self.stats[topic]['last_time'] = now
        elapsed = (now - self.stats[topic]['last_time']).to_sec()
        if elapsed > self.log_interval:
            with self.stats_lock:
                received = self.stats[topic]['received']
                processed = self.stats[topic]['processed']
                self.stats[topic]['received'] = 0
                self.stats[topic]['processed'] = 0
                self.stats[topic]['last_time'] = now
            if received > 0:
                rospy.loginfo(topic + ': processed %d out of %d in past %.1f sec (%.2f FPS)' % (processed, received, elapsed, processed / elapsed))
            else:
                rospy.loginfo(topic + ': no message received')

    def process(self, msg, topic):
        if msg.encoding == '8UC1' or msg.encoding == 'mono8':
            image = self.cv_bridge.imgmsg_to_cv2(msg)
            image_gray = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # print('8UC1')
        else:
            image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print('bgr8')
        # boxes = self.detectortrt.realtime_detect(cv_image)
        start = time.time()
        boxes = self.net.realtime_detect(image_gray)
        # features = self.net.infer(image_gray)
        stop = time.time()
        rospy.logdebug(topic + ': %.2f ' % (
                (stop - start) * 1000,
            ))
        # print((stop - start) * 1000)
       
        res = {'features': boxes, 'header': msg.header, 'topic': topic, 'image': image}
        self.result_queue.put(res)
            # print(features)

# def draw_keypoints(image, keypoints, scores):
#     upper_score = 0.5
#     lower_score = 0.1
#     scale = 1 / (upper_score - lower_score)
#     for p,s in zip(keypoints, scores):
#         s = min(max(s - lower_score, 0) * scale, 1)
#         color = (255 * (1 - s), 255 * (1 - s), 255) # BGR
#         cv2.circle(image, tuple(p), 3, color, 2)

def features_to_ros_msg(features, header):
    boxes_msg = ImageObj()
    boxes_msg.header = header

    boxes_tmp = []
        # print(boxes)

    if(len(features[0])):
            for i in range(len(features[0])):
                box_msg = ImageBox()
                box_msg.box = features[0][i]
                boxes_tmp.append(box_msg)

    boxes_msg.boxes = boxes_tmp
    # msg = ImageFeatures()
    # msg.header = header
    # msg.global_descriptor = features[0].tolist()
    return boxes_msg

if __name__ == "__main__":
    main()
