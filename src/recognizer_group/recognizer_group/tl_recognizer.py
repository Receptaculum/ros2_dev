import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer

import ultralytics
from ultralytics import YOLO

import cv2
import cv_bridge

import torch
import numpy as np

import os


## < Parameter> #####################################################################################

# 노드 이름
NODE_NAME = "tl_recognizer"

# 발행 토픽 이름
TOPIC_NAME = "tl_recognizer"

# 구독 토픽 이름 | 신호등 위치, 영상
SUB_TOPIC_NAME = ["cv_tl", "camera_publisher"]

# 영상 크기 (가로, 세로)
FRAME_SIZE = [640, 480]

# CV 처리 영상 출력 여부
DEBUG = True

######################################################################################################


## <로그 출력> #########################################################################################
# DEBUG	self.get_logger().debug("msg")
# INFO	self.get_logger().info("msg")
# WARN	self.get_logger().warn("msg")
# ERROR	self.get_logger().error("msg")
# FATAL	self.get_logger().fatal("msg")
#######################################################################################################


## <QOS> ##############################################################################################
# Reliability : RELIABLE(신뢰성 보장), BEST_EFFORT(손실 감수, 최대한 빠른 전송)
# Durability : VOLATILE(전달한 메시지 제거), TRANSIENT_LOCAL(전달한 메시지 유지) / (Subscriber가 없을 때에 한함)
# History : KEEP_LAST(depth 만큼의 메시지 유지), KEEP_ALL(모두 유지)
# Liveliness : 활성 상태 감시
# Deadline : 최소 동작 보장 
#######################################################################################################


class tl_recognizer(Node):
    def __init__(self, node_name, sub_topic_name, frame_size, debug):
        super().__init__(node_name)

        self.qos_pub = QoSProfile( # Publisher QOS 설정
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
                )

        self.qos_sub = QoSProfile( # Subscriber QOS 설정
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
                )
        
        self.frame_size = frame_size
        
        self.debug = debug

        self.subscriber_tl_location = self.create_subscription(Float32MultiArray, sub_topic_name[0], self.tl_location_callback, self.qos_sub)
        self.subscriber_frame = self.create_subscription(Image, sub_topic_name[1], self.frame_callback, self.qos_sub)


    def frame_callback(self, msg):
        self.img = np.array(msg.data).reshape([self.frame_size[1], self.frame_size[0], -1])



    def tl_location_callback(self, msg):
        self.x1, self.y1, self.x2, self.y2 = map(int, msg.data)
        self.w = abs(self.x1 - self.x2)
        self.h = abs(self.y1 - self.y2)

        try:
            self.img_cropped = self.img[self.y1 : self.y1 + self.h, self.x1 : self.x1 + self.w]

            self.r_sig, self.y_sig, self.g_sig = self.img_splitter(self.img_cropped, self.w)
            self.color_detector(self.r_sig)

            if self.debug == True:
                cv2.imshow("TL_Image", self.r_sig)
                cv2.waitKey(5)

        except:
            self.state = None
            self.get_logger().warn("Unable to read frame")


    def img_splitter(self, img, width):
        r_sig = img[0 : -1, 0 : 0 + int(width/3)]
        y_sig = img[0 : -1, int(width/3) : int(width/3*2)]
        g_sig = img[0 : -1, int(width/3*2) : int(width)]

        return r_sig, y_sig, g_sig
    

    def color_detector(self, img):
        y_range, x_range, channel_range = img.shape
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        offset = np.array([[0, 128, 128]])


        for y in range(y_range):
            for x in range(x_range):
                l, a, b = (img_lab[y][x] - offset).tolist()[0]

                print(l, a, b)

def main():
    rclpy.init()
    tl_recognizer_node = tl_recognizer(NODE_NAME, SUB_TOPIC_NAME, FRAME_SIZE, DEBUG)
    rclpy.spin(tl_recognizer_node)

    tl_recognizer_node.destroy_node()
    rclpy.shutdown()

if __name__== "__main__":
    main()